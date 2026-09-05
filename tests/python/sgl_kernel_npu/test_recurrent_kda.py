"""Single-op test for recurrent_kda AscendC kernel.

Tests the migrated recurrent_kda kernel (direct-call) on NPU by:
1. Calling torch.ops.npu.recurrent_kda directly with raw gates
2. Calling the kda_target_verify_npu wrapper with raw gates
3. Comparing both against a PyTorch reference implementation of the
   gated delta rule recurrent update.
4. Multi-batch test with different batch sizes.
"""
import argparse

import sgl_kernel_npu  # noqa: F401  registers npu ops
import torch
import torch_npu  # noqa: F401
from sgl_kernel_npu.fla.kda_target_verify import kda_target_verify_npu

EPS = 1e-6


def reference_recurrent_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate_raw: torch.Tensor,
    beta_raw: torch.Tensor,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    scale: float,
    use_gate_in_kernel: bool,
    use_beta_sigmoid: bool,
    state_v_first: bool = True,
) -> torch.Tensor:
    """PyTorch reference for the recurrent KDA delta rule update.

    Shapes (BSND with B=1):
      q, k: [1, T, H, K]
      v:    [1, T, HV, V]
      gate: [1, T, HV, K]
      beta: [1, T, HV]
      initial_state: [pool, HV, V, K]
    """
    _, T, H, K = q.shape
    HV = v.shape[2]
    V = v.shape[3]
    batch = cu_seqlens.numel() - 1

    q_f = q.float().squeeze(0)
    k_f = k.float().squeeze(0)
    v_f = v.float().squeeze(0)
    g_f = gate_raw.float().squeeze(0)
    b_f = beta_raw.float().squeeze(0)

    out = torch.zeros(T, HV, V, dtype=v.dtype, device=v.device)

    for b in range(batch):
        seq0 = int(cu_seqlens[b].item())
        seq1 = int(cu_seqlens[b + 1].item())
        if seq1 <= seq0:
            continue

        for hv in range(HV):
            h_head = hv // (HV // H)
            A_log_hv = A_log[hv].float()
            dt_bias_hv = dt_bias[hv].float()

            # Resolve initial state slot
            if ssm_state_indices.dim() == 2:
                init_slot = int(ssm_state_indices[b, 0].item())
            else:
                init_slot = int(ssm_state_indices[seq0].item())

            state = initial_state[init_slot, hv].float().clone()

            for t in range(seq0, seq1):
                q_t = q_f[t, h_head, :]
                k_t = k_f[t, h_head, :]
                v_t = v_f[t, hv, :]
                gate_t = g_f[t, hv, :]
                beta_t = b_f[t, hv]

                q_norm = q_t / (torch.sqrt(torch.sum(q_t * q_t)) + EPS)
                k_norm = k_t / (torch.sqrt(torch.sum(k_t * k_t)) + EPS)
                q_scaled = q_norm * scale

                if use_gate_in_kernel:
                    gate_decay = torch.exp(
                        -torch.exp(A_log_hv) * torch.nn.functional.softplus(gate_t + dt_bias_hv)
                    )
                else:
                    gate_decay = torch.exp(gate_t)

                if use_beta_sigmoid:
                    beta_val = torch.sigmoid(beta_t)
                else:
                    beta_val = beta_t

                state = state * gate_decay
                delta = v_t - state @ k_norm
                delta = delta * beta_val
                state = state + delta.unsqueeze(-1) * k_norm.unsqueeze(0)

                out[t, hv, :] = state @ q_scaled

    return out.unsqueeze(0)


def make_inputs(
    batch: int,
    seq_len_per_batch: int,
    H: int,
    HV: int,
    K: int,
    V: int,
    device: torch.device,
    seed: int = 42,
):
    torch.manual_seed(seed)
    T = batch * seq_len_per_batch

    q = torch.randn(1, T, H, K, dtype=torch.bfloat16, device=device)
    k = torch.randn(1, T, H, K, dtype=torch.bfloat16, device=device)
    v = torch.randn(1, T, HV, V, dtype=torch.bfloat16, device=device)
    gate_raw = torch.randn(1, T, HV, K, dtype=torch.float32, device=device)
    beta_raw = torch.randn(1, T, HV, dtype=torch.float32, device=device)

    A_log = torch.randn(HV, dtype=torch.float32, device=device) * 0.5
    dt_bias = torch.randn(HV, K, dtype=torch.float32, device=device) * 0.1

    pool_size = batch
    initial_state = torch.randn(pool_size, HV, V, K, dtype=torch.bfloat16, device=device) * 0.1

    cu_seqlens = torch.arange(0, T + 1, seq_len_per_batch, dtype=torch.int32, device=device)

    base_slots = torch.arange(batch, dtype=torch.int32, device=device).unsqueeze(1)
    step_offsets = torch.zeros(seq_len_per_batch, dtype=torch.int32, device=device).unsqueeze(0)
    ssm_state_indices = base_slots.expand(batch, seq_len_per_batch).contiguous()

    return q, k, v, gate_raw, beta_raw, initial_state, cu_seqlens, ssm_state_indices, A_log, dt_bias


def test_direct_op(device: torch.device, atol: float = 0.5, rtol: float = 0.1):
    print("\n=== Test 1: Direct recurrent_kda op (use_gate_in_kernel=True) ===")
    batch, seq_len = 2, 4
    H, HV, K, V = 4, 4, 128, 128
    scale = K ** -0.5

    q, k_t, v, gate_raw, beta_raw, initial_state, cu_seqlens, ssm_state_indices, A_log, dt_bias = (
        make_inputs(batch, seq_len, H, HV, K, V, device)
    )

    ref_out = reference_recurrent_kda(
        q=q, k=k_t, v=v, gate_raw=gate_raw, beta_raw=beta_raw,
        initial_state=initial_state, cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        A_log=A_log, dt_bias=dt_bias,
        scale=scale, use_gate_in_kernel=True, use_beta_sigmoid=True,
    )

    initial_state_npu = initial_state.clone()
    npu_out = torch.ops.npu.recurrent_kda(
        q.contiguous(), k_t.contiguous(), v.contiguous(),
        gate_raw.contiguous(), beta_raw.contiguous(),
        initial_state_npu,
        cu_seqlens, ssm_state_indices,
        a_log=A_log, dt_bias=dt_bias,
        scale=scale,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        allow_neg_eigval=False,
        safe_gate=False, lower_bound=-5.0,
        state_v_first=True,
    )
    torch.npu.synchronize()

    diff = (npu_out.float() - ref_out.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    ref_abs = ref_out.float().abs().mean().item()

    print(f"  Config: batch={batch}, seq_len={seq_len}, H={H}, HV={HV}, K={K}, V={V}")
    print(f"  Output shape: {npu_out.shape}")
    print(f"  Max abs diff:  {max_diff:.6f}")
    print(f"  Mean abs diff: {mean_diff:.6f}")
    print(f"  Ref mean abs:  {ref_abs:.6f}")
    print(f"  Relative err:  {mean_diff / (ref_abs + 1e-8):.6f}")
    assert max_diff < atol, f"max_diff {max_diff} >= atol {atol}"
    print("  [PASS]")


def test_wrapper_raw_gates(device: torch.device, atol: float = 0.5, rtol: float = 0.1):
    print("\n=== Test 2: kda_target_verify_npu wrapper (raw gates, in-kernel activation) ===")
    batch, cache_steps = 2, 4
    H, HV, K, V = 4, 4, 128, 128
    scale = K ** -0.5
    T = batch * cache_steps

    q, k_t, v, gate_raw, beta_raw, initial_state_source, _, _, A_log, dt_bias = (
        make_inputs(batch, cache_steps, H, HV, K, V, device)
    )

    scratch_size = batch
    intermediate_states_buffer = torch.zeros(
        scratch_size, cache_steps, HV, V, K, dtype=torch.bfloat16, device=device
    )
    intermediate_state_indices = torch.arange(batch, dtype=torch.int32, device=device)
    initial_state_indices = torch.arange(batch, dtype=torch.int32, device=device)

    ref_out = reference_recurrent_kda(
        q=q, k=k_t, v=v, gate_raw=gate_raw, beta_raw=beta_raw,
        initial_state=initial_state_source,
        cu_seqlens=torch.arange(0, T + 1, cache_steps, dtype=torch.int32, device=device),
        ssm_state_indices=torch.arange(batch, dtype=torch.int32, device=device).unsqueeze(1).expand(batch, cache_steps),
        A_log=A_log, dt_bias=dt_bias,
        scale=scale, use_gate_in_kernel=True, use_beta_sigmoid=True,
    )

    npu_out = kda_target_verify_npu(
        A_log=A_log,
        dt_bias=dt_bias,
        q=q,
        k=k_t,
        v=v,
        a=gate_raw,
        b=beta_raw,
        initial_state_source=initial_state_source,
        initial_state_indices=initial_state_indices,
        intermediate_states_buffer=intermediate_states_buffer,
        intermediate_state_indices=intermediate_state_indices,
        cache_steps=cache_steps,
    )
    torch.npu.synchronize()

    diff = (npu_out.float() - ref_out.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    ref_abs = ref_out.float().abs().mean().item()

    print(f"  Config: batch={batch}, cache_steps={cache_steps}, H={H}, HV={HV}, K={K}, V={V}")
    print(f"  Output shape: {npu_out.shape}")
    print(f"  Max abs diff:  {max_diff:.6f}")
    print(f"  Mean abs diff: {mean_diff:.6f}")
    print(f"  Ref mean abs:  {ref_abs:.6f}")
    print(f"  Relative err:  {mean_diff / (ref_abs + 1e-8):.6f}")
    nonzero_count = (intermediate_states_buffer.abs() > 0).sum().item()
    print(f"  Intermediate buffer nonzero elements: {nonzero_count}")
    assert max_diff < atol, f"max_diff {max_diff} >= atol {atol}"
    print("  [PASS]")


def test_multi_batch(device: torch.device, atol: float = 0.5, rtol: float = 0.1):
    """Test multi-batch correctness — critical for concurrent requests."""
    print("\n=== Test 3: Multi-batch (batch=4) ===")
    batch, cache_steps = 4, 4
    H, HV, K, V = 4, 4, 128, 128
    scale = K ** -0.5
    T = batch * cache_steps

    q, k_t, v, gate_raw, beta_raw, initial_state_source, _, _, A_log, dt_bias = (
        make_inputs(batch, cache_steps, H, HV, K, V, device)
    )

    intermediate_states_buffer = torch.zeros(
        batch, cache_steps, HV, V, K, dtype=torch.bfloat16, device=device
    )
    intermediate_state_indices = torch.arange(batch, dtype=torch.int32, device=device)
    initial_state_indices = torch.arange(batch, dtype=torch.int32, device=device)

    ref_out = reference_recurrent_kda(
        q=q, k=k_t, v=v, gate_raw=gate_raw, beta_raw=beta_raw,
        initial_state=initial_state_source,
        cu_seqlens=torch.arange(0, T + 1, cache_steps, dtype=torch.int32, device=device),
        ssm_state_indices=torch.arange(batch, dtype=torch.int32, device=device).unsqueeze(1).expand(batch, cache_steps),
        A_log=A_log, dt_bias=dt_bias,
        scale=scale, use_gate_in_kernel=True, use_beta_sigmoid=True,
    )

    npu_out = kda_target_verify_npu(
        A_log=A_log,
        dt_bias=dt_bias,
        q=q,
        k=k_t,
        v=v,
        a=gate_raw,
        b=beta_raw,
        initial_state_source=initial_state_source,
        initial_state_indices=initial_state_indices,
        intermediate_states_buffer=intermediate_states_buffer,
        intermediate_state_indices=intermediate_state_indices,
        cache_steps=cache_steps,
    )
    torch.npu.synchronize()

    diff = (npu_out.float() - ref_out.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    ref_abs = ref_out.float().abs().mean().item()

    print(f"  Config: batch={batch}, cache_steps={cache_steps}, H={H}, HV={HV}, K={K}, V={V}")
    print(f"  Output shape: {npu_out.shape}")
    print(f"  Max abs diff:  {max_diff:.6f}")
    print(f"  Mean abs diff: {mean_diff:.6f}")
    print(f"  Ref mean abs:  {ref_abs:.6f}")
    print(f"  Relative err:  {mean_diff / (ref_abs + 1e-8):.6f}")

    # Per-batch error check
    for b in range(batch):
        b_start = b * cache_steps
        b_end = (b + 1) * cache_steps
        b_diff = diff[0, b_start:b_end].max().item()
        print(f"  Batch {b} max diff: {b_diff:.6f}")

    assert max_diff < atol, f"max_diff {max_diff} >= atol {atol}"
    print("  [PASS]")


def test_v256(device: torch.device, atol: float = 0.5, rtol: float = 0.1):
    print("\n=== Test 4: V=256 configuration ===")
    batch, seq_len = 2, 4
    H, HV, K, V = 4, 4, 128, 256
    scale = K ** -0.5

    q, k_t, v, gate_raw, beta_raw, initial_state, cu_seqlens, ssm_state_indices, A_log, dt_bias = (
        make_inputs(batch, seq_len, H, HV, K, V, device)
    )

    ref_out = reference_recurrent_kda(
        q=q, k=k_t, v=v, gate_raw=gate_raw, beta_raw=beta_raw,
        initial_state=initial_state, cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        A_log=A_log, dt_bias=dt_bias,
        scale=scale, use_gate_in_kernel=True, use_beta_sigmoid=True,
    )

    initial_state_npu = initial_state.clone()
    npu_out = torch.ops.npu.recurrent_kda(
        q.contiguous(), k_t.contiguous(), v.contiguous(),
        gate_raw.contiguous(), beta_raw.contiguous(),
        initial_state_npu,
        cu_seqlens, ssm_state_indices,
        a_log=A_log, dt_bias=dt_bias,
        scale=scale,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        allow_neg_eigval=False,
        safe_gate=False, lower_bound=-5.0,
        state_v_first=True,
    )
    torch.npu.synchronize()

    diff = (npu_out.float() - ref_out.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    ref_abs = ref_out.float().abs().mean().item()
    print(f"  Config: batch={batch}, seq_len={seq_len}, H={H}, HV={HV}, K={K}, V={V}")
    print(f"  Max abs diff:  {max_diff:.6f}")
    print(f"  Mean abs diff: {mean_diff:.6f}")
    print(f"  Ref mean abs:  {ref_abs:.6f}")
    assert max_diff < atol, f"max_diff {max_diff} >= atol {atol}"
    print("  [PASS]")


def main():
    parser = argparse.ArgumentParser(description="Single-op test for recurrent_kda")
    parser.add_argument("--atol", type=float, default=1.0)
    parser.add_argument("--rtol", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not hasattr(torch.ops.npu, "recurrent_kda"):
        raise SystemExit("torch.ops.npu.recurrent_kda is not registered. Build sgl-kernel-npu first.")

    if not hasattr(torch, "npu") or torch.npu.device_count() <= 0:
        raise SystemExit("NPU device is not available")

    device = torch.device("npu")
    torch.manual_seed(args.seed)

    test_direct_op(device, atol=args.atol, rtol=args.rtol)
    test_wrapper_raw_gates(device, atol=args.atol, rtol=args.rtol)
    test_multi_batch(device, atol=args.atol, rtol=args.rtol)
    test_v256(device, atol=args.atol, rtol=args.rtol)

    print("\n=== All recurrent_kda tests passed ===")


if __name__ == "__main__":
    main()
