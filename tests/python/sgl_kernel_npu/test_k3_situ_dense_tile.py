import pytest
import torch
import torch_npu  # noqa: F401
from sgl_kernel_npu.activation.situ import situ_and_mul


def _reference(x: torch.Tensor) -> torch.Tensor:
    gate, up = x.float().chunk(2, dim=-1)
    gate = 4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)
    up = 25.0 * torch.tanh(up / 25.0)
    return (gate * up).to(x.dtype)


@pytest.mark.parametrize(
    ("tokens", "intermediate"),
    [
        (1, 192),
        (32, 192),
        (256, 192),
        (257, 192),
        (256, 512),
        (256, 1056),
        (1, 3072),
        (16, 3072),
        (1, 33792),
        (16, 33792),
        (64, 33792),
    ],
)
@torch.no_grad()
def test_situ_dense_tile_matches_reference(tokens: int, intermediate: int):
    torch.manual_seed(20260820 + tokens)
    host = torch.randn(tokens, 2 * intermediate, dtype=torch.float32).clamp_(-8, 8)
    x = host.to(dtype=torch.bfloat16, device="npu")

    actual = situ_and_mul(x, beta=4.0, linear_beta=25.0)
    expected = _reference(host.to(torch.bfloat16))

    torch.testing.assert_close(actual.cpu(), expected, atol=3e-2, rtol=1e-2)


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@torch.no_grad()
def test_narrow_grouped_situ_still_obeys_valid_row_counts(dtype):
    x = torch.randn(8, 384, device="npu", dtype=torch.bfloat16)
    counts = torch.tensor([3, 2, 1], device="npu", dtype=dtype)
    actual = situ_and_mul(x, group_list=counts, group_list_type=1)
    torch.testing.assert_close(
        actual[:6].cpu(), _reference(x.cpu())[:6], atol=3e-2, rtol=1e-2
    )


@torch.no_grad()
def test_narrow_situ_graph_replay_reads_updated_input():
    x = torch.randn(256, 384, device="npu", dtype=torch.bfloat16)
    situ_and_mul(x)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        actual = situ_and_mul(x)
    for _ in range(3):
        x.copy_(torch.randn_like(x))
        expected = situ_and_mul(x)
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
