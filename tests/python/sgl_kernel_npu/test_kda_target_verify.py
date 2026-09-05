import pytest
import torch
from sgl_kernel_npu.fla.kda_gate import fused_kda_gate_npu
from sgl_kernel_npu.fla.kda_target_verify import kda_target_verify_npu


def _target_verify_cpu_reference(
    *,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    initial_state: torch.Tensor,
    initial_indices: torch.Tensor,
    snapshot_template: torch.Tensor,
    snapshot_indices: torch.Tensor,
    steps: int,
    lower_bound: float | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    q_cpu = q.detach().cpu().squeeze(0).float()
    k_cpu = k.detach().cpu().squeeze(0).float()
    v_cpu = v.detach().cpu().squeeze(0).float()
    a_cpu = a.detach().cpu().squeeze(0).float() if a.ndim == 4 else a.cpu().float()
    b_cpu = b.detach().cpu().squeeze(0).float() if b.ndim == 3 else b.cpu().float()
    A_cpu = A_log.detach().cpu().reshape(-1).float().exp()
    dt_cpu = dt_bias.detach().cpu().reshape(k.shape[2], k.shape[3]).float()
    state_source = initial_state.detach().cpu().float()
    initial_indices_cpu = initial_indices.detach().cpu()
    snapshot_indices_cpu = snapshot_indices.detach().cpu()
    expected_snapshots = snapshot_template.detach().cpu().clone()

    tokens = q.shape[1]
    batch = tokens // steps
    h_q, key_dim = q.shape[2:]
    h_k = k.shape[2]
    h_v, value_dim = v.shape[2:]
    q_ratio = h_v // h_q
    k_ratio = h_v // h_k
    output = torch.zeros(1, tokens, h_v, value_dim, dtype=v.dtype)
    scale = key_dim**-0.5

    for batch_idx in range(batch):
        initial_idx = int(initial_indices_cpu[batch_idx])
        if initial_idx < 0:
            continue
        snapshot_idx = int(snapshot_indices_cpu[batch_idx])
        state = state_source[initial_idx].clone()
        for step in range(steps):
            token = batch_idx * steps + step
            for value_head in range(h_v):
                query_head = value_head // q_ratio
                key_head = value_head // k_ratio
                query = q_cpu[token, query_head]
                key = k_cpu[token, key_head]
                query = query / (query.square().sum().sqrt() + 1e-6)
                key = key / (key.square().sum().sqrt() + 1e-6)
                query = query * scale
                gate_input = a_cpu[token, key_head] + dt_cpu[key_head]
                if lower_bound is None:
                    log_gate = -A_cpu[key_head] * torch.nn.functional.softplus(
                        gate_input
                    )
                else:
                    log_gate = lower_bound * torch.sigmoid(A_cpu[key_head] * gate_input)
                decay = log_gate.exp()
                beta = b_cpu[token, value_head].sigmoid()
                value = v_cpu[token, value_head].clone()
                head_state = state[value_head]
                head_state *= decay.unsqueeze(0)
                value -= (head_state * key.unsqueeze(0)).sum(dim=1)
                value *= beta
                head_state += value.unsqueeze(1) * key.unsqueeze(0)
                output[0, token, value_head] = (
                    (head_state * query.unsqueeze(0)).sum(dim=1).to(output.dtype)
                )
            if snapshot_idx >= 0:
                expected_snapshots[snapshot_idx, step] = state.to(
                    expected_snapshots.dtype
                )

    return output, expected_snapshots


def test_kda_target_verify_raw_gates_match_preactivated_gates():
    device = torch.device("npu")
    batch, steps, heads, key_dim, value_dim = 2, 3, 2, 8, 8
    tokens = batch * steps
    q = torch.randn(1, tokens, heads, key_dim, dtype=torch.bfloat16, device=device)
    k = torch.randn_like(q)
    v = torch.randn(1, tokens, heads, value_dim, dtype=torch.bfloat16, device=device)
    raw_a = torch.randn_like(q)
    raw_b = torch.randn(1, tokens, heads, dtype=torch.bfloat16, device=device)
    A_log = torch.randn(1, 1, heads, 1, dtype=torch.float32, device=device)
    dt_bias = torch.randn(heads * key_dim, dtype=torch.float32, device=device)
    assert A_log.shape == (1, 1, heads, 1)
    assert dt_bias.shape == (heads * key_dim,)
    initial_state = torch.randn(
        batch, heads, value_dim, key_dim, dtype=torch.bfloat16, device=device
    )
    initial_indices = torch.arange(batch, dtype=torch.int32, device=device)
    intermediate_indices = torch.arange(batch, dtype=torch.int32, device=device)
    raw_scratch = torch.empty(
        batch,
        steps,
        heads,
        value_dim,
        key_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    preactivated_scratch = torch.empty_like(raw_scratch)
    lower_bound = -5.0

    raw_output = kda_target_verify_npu(
        A_log=A_log,
        dt_bias=dt_bias,
        q=q,
        k=k,
        v=v,
        a=raw_a,
        b=raw_b,
        initial_state_source=initial_state,
        initial_state_indices=initial_indices,
        intermediate_states_buffer=raw_scratch,
        intermediate_state_indices=intermediate_indices,
        cache_steps=steps,
        gates_are_preactivated=False,
        lower_bound=lower_bound,
    )
    preactivated_a = fused_kda_gate_npu(
        raw_a.flatten(-2),
        A_log,
        key_dim,
        gate_bias=dt_bias,
        lower_bound=lower_bound,
    )
    preactivated_b = raw_b.float().sigmoid()
    preactivated_output = kda_target_verify_npu(
        A_log=A_log,
        dt_bias=dt_bias,
        q=q,
        k=k,
        v=v,
        a=preactivated_a,
        b=preactivated_b,
        initial_state_source=initial_state,
        initial_state_indices=initial_indices,
        intermediate_states_buffer=preactivated_scratch,
        intermediate_state_indices=intermediate_indices,
        cache_steps=steps,
        gates_are_preactivated=True,
    )
    torch.testing.assert_close(
        raw_output.float(), preactivated_output.float(), rtol=2e-2, atol=2e-2
    )
    torch.testing.assert_close(
        raw_scratch.float(), preactivated_scratch.float(), rtol=2e-2, atol=2e-2
    )

    with pytest.raises(
        ValueError, match="lower_bound must already be reflected in preactivated gates"
    ):
        kda_target_verify_npu(
            A_log=A_log,
            dt_bias=dt_bias,
            q=q,
            k=k,
            v=v,
            a=preactivated_a,
            b=preactivated_b,
            initial_state_source=initial_state,
            initial_state_indices=initial_indices,
            intermediate_states_buffer=preactivated_scratch,
            intermediate_state_indices=intermediate_indices,
            cache_steps=steps,
            gates_are_preactivated=True,
            lower_bound=lower_bound,
        )


def test_kda_target_verify_padding_matches_cpu_and_preserves_snapshot():
    device = torch.device("npu")
    batch, steps, heads, key_dim, value_dim = 3, 4, 2, 8, 8
    tokens = batch * steps
    q = torch.randn(1, tokens, heads, key_dim, dtype=torch.bfloat16, device=device)
    k = torch.randn_like(q)
    v = torch.randn(1, tokens, heads, value_dim, dtype=torch.bfloat16, device=device)
    raw_a = torch.randn_like(q)
    raw_b = torch.randn(1, tokens, heads, dtype=torch.bfloat16, device=device)
    # Deliberately poison the padded producer rows. A program-level skip must
    # still emit exact zeros without propagating NaNs into output or snapshots.
    q[:, steps : 2 * steps].fill_(float("nan"))
    k[:, steps : 2 * steps].fill_(float("nan"))
    v[:, steps : 2 * steps].fill_(float("nan"))
    raw_a[:, steps : 2 * steps].fill_(float("nan"))
    raw_b[:, steps : 2 * steps].fill_(float("nan"))

    A_log = torch.randn(1, 1, heads, 1, dtype=torch.float32, device=device)
    dt_bias = torch.randn(heads * key_dim, dtype=torch.float32, device=device)
    initial_state = torch.randn(
        2, heads, value_dim, key_dim, dtype=torch.bfloat16, device=device
    )
    initial_indices = torch.tensor([0, -1, 1], dtype=torch.int32, device=device)
    intermediate_indices = torch.arange(batch, dtype=torch.int32, device=device)
    scratch = torch.full(
        (batch, steps, heads, value_dim, key_dim),
        3.0,
        dtype=torch.bfloat16,
        device=device,
    )
    scratch_before = scratch.clone()
    lower_bound = -5.0
    expected_output, expected_scratch = _target_verify_cpu_reference(
        A_log=A_log,
        dt_bias=dt_bias,
        q=q,
        k=k,
        v=v,
        a=raw_a,
        b=raw_b,
        initial_state=initial_state,
        initial_indices=initial_indices,
        snapshot_template=scratch,
        snapshot_indices=intermediate_indices,
        steps=steps,
        lower_bound=lower_bound,
    )

    actual = kda_target_verify_npu(
        A_log=A_log,
        dt_bias=dt_bias,
        q=q,
        k=k,
        v=v,
        a=raw_a,
        b=raw_b,
        initial_state_source=initial_state,
        initial_state_indices=initial_indices,
        intermediate_states_buffer=scratch,
        intermediate_state_indices=intermediate_indices,
        cache_steps=steps,
        gates_are_preactivated=False,
        lower_bound=lower_bound,
    )

    torch.testing.assert_close(
        actual.cpu().float(), expected_output.float(), atol=2e-2, rtol=2e-2
    )
    torch.testing.assert_close(
        scratch.cpu().float(), expected_scratch.float(), atol=2e-2, rtol=2e-2
    )
    torch.testing.assert_close(
        actual[:, steps : 2 * steps],
        torch.zeros_like(actual[:, steps : 2 * steps]),
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(scratch[1], scratch_before[1], atol=0, rtol=0)
