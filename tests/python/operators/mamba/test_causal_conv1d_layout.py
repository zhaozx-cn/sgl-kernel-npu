"""Prefill output layouts, state equivalence, and unchanged update semantics."""

import pytest
import torch


@pytest.fixture(params=["torch_native", "ascendc"])
def implementation(request):
    return request.param


def make_inputs(device, dtype, channels, width, dense=False, length=9):
    generator = torch.Generator().manual_seed(20261123)

    def rand(shape):
        return (torch.randn(shape, generator=generator, device="cpu") * 0.3).to(dtype).to(device)

    lengths = [length, length, length] if dense else [2, 0, length - 3, 1]
    boundaries = [0]
    for size in lengths:
        boundaries.append(boundaries[-1] + size)
    shape = (3, length, channels) if dense else (length, channels)
    indices = [0, -1, 4] if dense else [0, 2, 4, -1]
    live = torch.ones(shape[:-1], dtype=torch.bool, device="cpu")
    if dense:
        live[1] = False
    else:
        live[-1] = False
    return dict(
        x=rand(shape),
        weight=rand((width, channels)),
        conv_states=rand((6, width + 2, channels)),
        bias=rand((channels,)) if width != 2 else None,
        query_start_loc=None if dense else torch.tensor(boundaries, dtype=torch.int64, device=device),
        cache_indices=torch.tensor(indices, dtype=torch.int64, device=device),
        has_initial_state=torch.tensor([True, False, True] if dense else [True, False, False, False], device=device),
        activation_mode=int(width != 2),
    ), live


def clone_inputs(data, device=None):
    return {
        name: value.clone().to(device or value.device) if isinstance(value, torch.Tensor) else value
        for name, value in data.items()
    }


def token_order(output, x):
    if x.ndim == 2:
        return output.permute(1, 0, 2).reshape(x.shape)
    return output.permute(0, 2, 1, 3).reshape(x.shape)


@pytest.mark.parametrize(
    "dense,channels,heads,width,dtype,length",
    [
        (False, 384, 3, 2, torch.float16, 9),
        (False, 2048, 16, 3, torch.bfloat16, 129),
        (False, 12288, 96, 4, torch.bfloat16, 257),
        (False, 4800, 25, 4, torch.float16, 65),  # a 4096-channel tile cuts a 192-wide head
        (False, 96, 6, 3, torch.bfloat16, 9),
        (False, 4224, 1, 2, torch.float16, 9),  # one head spans channel tiles
        (True, 384, 3, 2, torch.bfloat16, 1),
        (True, 2048, 16, 3, torch.float16, 17),
        (True, 4800, 25, 4, torch.bfloat16, 65),
    ],
)
def test_prefill_layout_and_state(implementation, dense, channels, heads, width, dtype, length):
    from sgl_kernel_npu.operators.mamba import causal_conv1d

    device = "npu" if implementation == "ascendc" else "cpu"
    original, live = make_inputs(device, dtype, channels, width, dense, length)
    token, head = clone_inputs(original), clone_inputs(original)
    reference = clone_inputs(original, "cpu")
    expected = causal_conv1d(**reference, impl="torch_native")
    y_token = causal_conv1d(**token, impl=implementation)
    y_head = causal_conv1d(**head, output_layout="HEAD_FIRST", head_num=heads, impl=implementation)
    expected_shape = (3, heads, length, channels // heads) if dense else (heads, length, channels // heads)
    assert y_head.shape == expected_shape
    assert y_head.is_contiguous()
    restored = token_order(y_head, original["x"]).cpu()
    torch.testing.assert_close(restored[live], y_token.cpu()[live], rtol=0, atol=0)
    torch.testing.assert_close(restored[live], expected[live], rtol=0.03, atol=0.004)
    torch.testing.assert_close(head["conv_states"].cpu(), reference["conv_states"], rtol=0, atol=0)
    torch.testing.assert_close(token["conv_states"], head["conv_states"], rtol=0, atol=0)
    for name, value in original.items():
        if isinstance(value, torch.Tensor) and name != "conv_states":
            assert torch.equal(value, head[name]), name
    if not dense and heads >= 3:
        # Packed B=1 Q/K/V groups can be passed as contiguous views.
        q, k, v = y_head.split([1, 1, heads - 2], dim=0)
        assert all(part.is_contiguous() for part in (q, k, v))


@pytest.mark.parametrize("accepted", [False, True])
def test_update_remains_token_first(implementation, accepted):
    from sgl_kernel_npu.operators.mamba import causal_conv1d

    device = "npu" if implementation == "ascendc" else "cpu"
    inputs, live = make_inputs(device, torch.bfloat16, 384, 4, dense=True, length=3 if accepted else 1)
    inputs["run_mode"] = 1
    if accepted:
        inputs["num_accepted_tokens"] = torch.tensor([1, 2, 3], dtype=torch.int32, device=device)
    reference = clone_inputs(inputs, "cpu")
    expected = causal_conv1d(**reference, impl="torch_native")
    result = causal_conv1d(**inputs, impl=implementation)
    assert result.shape == inputs["x"].shape
    torch.testing.assert_close(result.cpu()[live], expected[live], rtol=0.03, atol=0.004)
    torch.testing.assert_close(inputs["conv_states"].cpu(), reference["conv_states"], rtol=0, atol=0)


@pytest.mark.parametrize("kwargs", [
    {"output_layout": "INVALID"},
    {"output_layout": "TOKEN_FIRST", "head_num": 3},
    {"output_layout": "HEAD_FIRST", "head_num": 0},
    {"output_layout": "HEAD_FIRST", "head_num": -1},
    {"output_layout": "HEAD_FIRST", "head_num": 5},
    {"output_layout": "HEAD_FIRST", "head_num": 128},
    {"output_layout": "HEAD_FIRST", "head_num": 3, "run_mode": 1},
    {"output_layout": "HEAD_FIRST", "head_num": 3, "run_mode": 2},
])
def test_invalid_layout_is_rejected_before_state_update(implementation, kwargs):
    from sgl_kernel_npu.operators.mamba import causal_conv1d

    device = "npu" if implementation == "ascendc" else "cpu"
    inputs, _ = make_inputs(device, torch.bfloat16, 384, 4)
    state = inputs["conv_states"].clone()
    with pytest.raises((RuntimeError, ValueError)):
        causal_conv1d(**inputs, **kwargs, impl=implementation)
    assert torch.equal(inputs["conv_states"], state)


def test_layout_and_head_count_in_graph_cache():
    from sgl_kernel_npu.operators.mamba import causal_conv1d

    inputs, live = make_inputs("npu", torch.bfloat16, 384, 4)
    inputs["cache_indices"][-1] = 5
    live[:] = True
    variants = [{}, {"output_layout": "HEAD_FIRST", "head_num": 3},
                {"output_layout": "HEAD_FIRST", "head_num": 6}]
    calls = [clone_inputs(inputs) for _ in variants]
    for _ in range(3):
        for data, options in zip(calls, variants):
            causal_conv1d(**data, **options)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        outputs = [causal_conv1d(**data, **options) for data, options in zip(calls, variants)]
    for iteration in range(2):
        for data in calls:
            data["conv_states"].copy_(inputs["conv_states"])
            data["x"].copy_(inputs["x"] * (iteration + 1))
            if iteration:
                data["query_start_loc"].copy_(torch.tensor([0, 1, 2, 8, 9], device="npu"))
        references = [clone_inputs(data, "cpu") for data in calls]
        graph.replay()
        torch.npu.synchronize()
        for data, output, reference, options in zip(calls, outputs, references, variants):
            expected = causal_conv1d(**reference, impl="torch_native")
            actual = token_order(output, data["x"]) if options else output
            torch.testing.assert_close(actual.cpu(), expected, rtol=0.03, atol=0.004)
            torch.testing.assert_close(data["conv_states"].cpu(), reference["conv_states"], rtol=0, atol=0)
