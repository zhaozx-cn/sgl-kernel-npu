from typing import Optional

import torch
import triton
import triton.language as tl

try:
    import triton.language.extra.cann.extension as cann_extension
except ImportError:
    cann_extension = None


@triton.jit
def _kda_target_verify_kernel(
    A_log_ptr,
    dt_bias_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    a_ptr,
    b_ptr,
    initial_state_ptr,
    initial_indices_ptr,
    snapshot_ptr,
    snapshot_indices_ptr,
    out_ptr,
    scale,
    lower_bound,
    stride_q_token: tl.constexpr,
    stride_q_head: tl.constexpr,
    stride_q_dim: tl.constexpr,
    stride_k_token: tl.constexpr,
    stride_k_head: tl.constexpr,
    stride_k_dim: tl.constexpr,
    stride_v_token: tl.constexpr,
    stride_v_head: tl.constexpr,
    stride_v_dim: tl.constexpr,
    stride_a_token: tl.constexpr,
    stride_a_head: tl.constexpr,
    stride_a_dim: tl.constexpr,
    stride_b_token: tl.constexpr,
    stride_b_head: tl.constexpr,
    initial_stride_0,
    initial_stride_1,
    initial_stride_2,
    initial_stride_3,
    snapshot_stride_0,
    snapshot_stride_1,
    snapshot_stride_2,
    snapshot_stride_3,
    snapshot_stride_4,
    H_Q: tl.constexpr,
    H_K: tl.constexpr,
    H_V: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    STEPS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    GATES_ARE_PREACTIVATED: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr,
    PRECOMPUTE_RAW_GATES: tl.constexpr,
    BT: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_hv = tl.program_id(1)
    pid_v = tl.program_id(2)

    offset_k = tl.arange(0, BK)
    offset_v = pid_v * BV + tl.arange(0, BV)
    mask_k = offset_k < K
    mask_v = offset_v < V
    mask_state = mask_v[:, None] & mask_k[None, :]

    q_ratio = H_V // H_Q
    k_ratio = H_V // H_K
    q_head = pid_hv // q_ratio
    k_head = pid_hv // k_ratio
    initial_idx = tl.load(initial_indices_ptr + pid_batch).to(tl.int64)
    snapshot_idx = tl.load(snapshot_indices_ptr + pid_batch).to(tl.int64)

    # CUDA/NPU graph padding uses -1 as the persistent-state sentinel.  The
    # causal-conv producer skips those requests, so its corresponding q/k/v
    # rows are intentionally undefined.  Do not feed them through the
    # recurrence (or write bogus snapshots); explicitly zero the output rows
    # because later dense layers still consume the full captured batch.
    if initial_idx < 0:
        for step in tl.static_range(0, STEPS):
            token = pid_batch * STEPS + step
            tl.store(
                out_ptr + (token * H_V + pid_hv) * V + offset_v,
                0.0,
                mask=mask_v,
            )
        return

    if PRECOMPUTE_RAW_GATES:
        # Gate activation has no state dependency. Vectorize all verify tokens
        # before loading the large recurrent state tile; keep only the final
        # FP32 decay/beta values live across the sequential state updates.
        gate_steps = tl.arange(0, BT)
        gate_tokens = pid_batch * STEPS + gate_steps
        # Load packed/strided gates as one vector before restoring the token
        # axis. Narrow rows must not become sub-block 2D gather operations.
        gate_offsets = tl.arange(0, BT * BK)
        gate_token_offsets = pid_batch * STEPS + gate_offsets // BK
        gate_key_offsets = gate_offsets % BK
        raw_gate = (
            tl.load(
                a_ptr
                + gate_token_offsets * stride_a_token
                + k_head * stride_a_head
                + gate_key_offsets * stride_a_dim,
                mask=(gate_offsets // BK < STEPS) & (gate_key_offsets < K),
                other=0.0,
            )
            .to(tl.float32)
            .reshape((BT, BK))
        )
        gate_bias = tl.load(
            dt_bias_ptr + k_head * K + offset_k, mask=mask_k, other=0.0
        ).to(tl.float32)
        gate_exp_A = tl.exp(tl.load(A_log_ptr + k_head).to(tl.float32))
        gate_input_all = raw_gate + gate_bias[None, :]
        if USE_LOWER_BOUND:
            log_gate_all = lower_bound * tl.sigmoid(gate_exp_A * gate_input_all)
        else:
            softplus_all = tl.where(
                gate_input_all <= 20.0,
                tl.log(1.0 + tl.exp(gate_input_all)),
                gate_input_all,
            )
            log_gate_all = -gate_exp_A * softplus_all
        decay_all = tl.exp(log_gate_all)
        beta_raw_all = tl.load(
            b_ptr + gate_tokens * stride_b_token + pid_hv * stride_b_head,
            mask=gate_steps < STEPS,
            other=0.0,
        ).to(tl.float32)
        beta_all = 1.0 / (1.0 + tl.exp(-beta_raw_all))

    initial_offsets = (
        initial_idx * initial_stride_0
        + pid_hv * initial_stride_1
        + offset_v[:, None] * initial_stride_2
        + offset_k[None, :] * initial_stride_3
    )
    state = tl.load(
        initial_state_ptr + initial_offsets,
        mask=(initial_idx >= 0) & mask_state,
        other=0.0,
    ).to(tl.float32)

    exp_A = tl.zeros((), dtype=tl.float32)
    dt_bias = tl.zeros((BK,), dtype=tl.float32)
    if not GATES_ARE_PREACTIVATED and not PRECOMPUTE_RAW_GATES:
        exp_A = tl.exp(tl.load(A_log_ptr + k_head).to(tl.float32))
        dt_bias = tl.load(
            dt_bias_ptr + k_head * K + offset_k,
            mask=mask_k,
            other=0.0,
        ).to(tl.float32)

    for step in tl.static_range(0, STEPS):
        token = pid_batch * STEPS + step
        q = tl.load(
            q_ptr
            + token * stride_q_token
            + q_head * stride_q_head
            + offset_k * stride_q_dim,
            mask=mask_k,
            other=0.0,
        ).to(tl.float32)
        k = tl.load(
            k_ptr
            + token * stride_k_token
            + k_head * stride_k_head
            + offset_k * stride_k_dim,
            mask=mask_k,
            other=0.0,
        ).to(tl.float32)
        value = tl.load(
            v_ptr
            + token * stride_v_token
            + pid_hv * stride_v_head
            + offset_v * stride_v_dim,
            mask=mask_v,
            other=0.0,
        ).to(tl.float32)
        if not PRECOMPUTE_RAW_GATES:
            a = tl.load(
                a_ptr
                + token * stride_a_token
                + k_head * stride_a_head
                + offset_k * stride_a_dim,
                mask=mask_k,
                other=0.0,
            ).to(tl.float32)
            beta_input = tl.load(
                b_ptr + token * stride_b_token + pid_hv * stride_b_head
            ).to(tl.float32)

        q = q / (tl.sqrt(tl.sum(q * q, axis=0)) + 1e-6)
        k = k / (tl.sqrt(tl.sum(k * k, axis=0)) + 1e-6)
        q *= scale

        if PRECOMPUTE_RAW_GATES:
            # Static slices avoid a generic gather/reduction in each step.
            gate = cann_extension.extract_slice(
                decay_all, offsets=(step, 0), sizes=(1, BK), strides=(1, 1)
            ).reshape((BK,))
            # A one-element reduction produces a scalar without constructing
            # the zero-dimensional block type forbidden by Triton reshape.
            beta = tl.sum(
                cann_extension.extract_slice(
                    beta_all, offsets=(step,), sizes=(1,), strides=(1,)
                ),
                axis=0,
            )
        elif GATES_ARE_PREACTIVATED:
            gate = tl.exp(a)
            beta = beta_input
        else:
            gate_input = a + dt_bias
            if USE_LOWER_BOUND:
                log_gate = lower_bound * tl.sigmoid(exp_A * gate_input)
            else:
                softplus = tl.where(
                    gate_input <= 20.0,
                    tl.log(1.0 + tl.exp(gate_input)),
                    gate_input,
                )
                log_gate = -exp_A * softplus
            gate = tl.exp(log_gate)
            beta = 1.0 / (1.0 + tl.exp(-beta_input))

        state *= gate[None, :]
        value -= tl.sum(state * k[None, :], axis=1)
        value *= beta
        state += value[:, None] * k[None, :]
        output = tl.sum(state * q[None, :], axis=1)

        tl.store(
            out_ptr + (token * H_V + pid_hv) * V + offset_v,
            output,
            mask=mask_v,
        )
        snapshot_offsets = (
            snapshot_idx * snapshot_stride_0
            + step * snapshot_stride_1
            + pid_hv * snapshot_stride_2
            + offset_v[:, None] * snapshot_stride_3
            + offset_k[None, :] * snapshot_stride_4
        )
        tl.store(
            snapshot_ptr + snapshot_offsets,
            state,
            mask=(snapshot_idx >= 0) & mask_state,
        )


def kda_target_verify_npu(
    *,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    initial_state_source: torch.Tensor,
    initial_state_indices: torch.Tensor,
    intermediate_states_buffer: torch.Tensor,
    intermediate_state_indices: torch.Tensor,
    cache_steps: int,
    scale: Optional[float] = None,
    gates_are_preactivated: Optional[bool] = None,
    lower_bound: Optional[float] = None,
    precompute_raw_gates: bool = False,
    value_block_size: Optional[int] = None,
) -> torch.Tensor:
    """KDA fixed-width target verification with per-step state snapshots.

    The persistent and intermediate state layout is the Ascend KDA layout
    ``[..., H_v, V, K]``. The persistent cache is read-only.

    When ``gates_are_preactivated`` is true, ``a`` is the already computed
    log-decay and ``b`` is already sigmoid activated. Otherwise both tensors
    are raw: ``lower_bound`` selects the K3 safe gate
    ``lower_bound * sigmoid(exp(A_log) * (a + dt_bias))``; when it is absent,
    the log-decay is ``-exp(A_log) * softplus(a + dt_bias)``. Both gate tensors
    may include the SGLang leading singleton. When the flag is omitted, a
    paired leading singleton selects the preactivated mode.

    ``precompute_raw_gates`` is an opt-in experiment for at most 16 verify
    steps: activate gates over the token axis before the recurrent loop.
    ``value_block_size`` optionally overrides the V tile (32, 64, or 128).
    Defaults retain the original dispatch; benchmark on the target NPU before
    enabling either option in serving.
    """
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, and v must have shape [1, tokens, heads, dim]")
    if q.shape[0] != 1 or k.shape[0] != 1 or v.shape[0] != 1:
        raise ValueError("the leading q, k, and v dimension must be one")
    if cache_steps <= 0 or q.shape[1] % cache_steps != 0:
        raise ValueError("tokens must be divisible by positive cache_steps")
    if q.shape[1] != k.shape[1] or q.shape[1] != v.shape[1]:
        raise ValueError("q, k, and v token dimensions must match")

    batch = q.shape[1] // cache_steps
    h_q, key_dim = q.shape[2:]
    h_k = k.shape[2]
    h_v, value_dim = v.shape[2:]
    a_has_leading_singleton = a.ndim == 4
    b_has_leading_singleton = b.ndim == 3
    if a_has_leading_singleton != b_has_leading_singleton:
        raise ValueError("a and b must use the leading singleton together")
    if gates_are_preactivated is None:
        gates_are_preactivated = a_has_leading_singleton
    if a.ndim == 4:
        if a.shape[0] != 1:
            raise ValueError("4D a must have a leading singleton dimension")
        a = a.squeeze(0)
    if b.ndim == 3:
        if b.shape[0] != 1:
            raise ValueError("3D b must have a leading singleton dimension")
        b = b.squeeze(0)
    if k.shape[3] != key_dim:
        raise ValueError("q and k key dimensions must match")
    if h_v % h_q != 0 or h_v % h_k != 0:
        raise ValueError("value heads must be divisible by q and k heads")
    if tuple(a.shape) != (q.shape[1], h_k, key_dim):
        raise ValueError("a must have shape [tokens, H_k, K]")
    if tuple(b.shape) != (q.shape[1], h_v):
        raise ValueError("b must have shape [tokens, H_v]")
    # K3 stores A_log as [1, 1, H_k, 1] and dt_bias as [H_k * K].
    # The kernel reads both as flat buffers, so validate element counts.
    if not gates_are_preactivated and (
        A_log.numel() != h_k or dt_bias.numel() != h_k * key_dim
    ):
        raise ValueError("A_log and dt_bias shapes do not match KDA heads")
    if initial_state_source.ndim != 4 or tuple(initial_state_source.shape[1:]) != (
        h_v,
        value_dim,
        key_dim,
    ):
        raise ValueError("initial state must have shape [pool, H_v, V, K]")
    if intermediate_states_buffer.ndim != 5 or tuple(
        intermediate_states_buffer.shape[1:]
    ) != (cache_steps, h_v, value_dim, key_dim):
        raise ValueError("intermediate state must have shape [scratch, T, H_v, V, K]")
    if initial_state_indices.ndim != 1 or initial_state_indices.numel() < batch:
        raise ValueError("initial_state_indices must contain at least B entries")
    if (
        intermediate_state_indices.ndim != 1
        or intermediate_state_indices.numel() < batch
    ):
        raise ValueError("intermediate_state_indices must contain at least B entries")

    # SGLang produces q/k/v as views of a packed QKV tensor. The kernel consumes
    # explicit strides so serving can avoid five per-layer materializations.
    tensors = [
        A_log,
        dt_bias,
        q,
        k,
        v,
        a,
        b,
        initial_state_source,
        initial_state_indices,
        intermediate_states_buffer,
        intermediate_state_indices,
    ]
    if any(t.device != q.device for t in tensors):
        raise ValueError("all tensors must be on the same device")
    A_log = A_log.contiguous()
    dt_bias = dt_bias.contiguous()
    initial_state_indices = initial_state_indices.contiguous()
    intermediate_state_indices = intermediate_state_indices.contiguous()
    if initial_state_source.dtype != intermediate_states_buffer.dtype:
        raise ValueError("persistent and intermediate state dtypes must match")
    if initial_state_indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("initial_state_indices must be int32 or int64")
    if intermediate_state_indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("intermediate_state_indices must be int32 or int64")

    if scale is None:
        scale = key_dim**-0.5
    if scale <= 0:
        raise ValueError("scale must be positive")
    if gates_are_preactivated and lower_bound is not None:
        raise ValueError("lower_bound must already be reflected in preactivated gates")
    if precompute_raw_gates:
        if gates_are_preactivated:
            raise ValueError("precompute_raw_gates requires raw gates")
        if cache_steps > 16:
            raise ValueError("precompute_raw_gates supports at most 16 verify steps")
        if cann_extension is None or not hasattr(cann_extension, "extract_slice"):
            raise RuntimeError(
                "precompute_raw_gates requires Triton-Ascend cann.extract_slice"
            )
    if value_block_size not in (None, 32, 64, 128):
        raise ValueError("value_block_size must be None, 32, 64, or 128")

    out = torch.empty((1, q.shape[1], h_v, value_dim), dtype=v.dtype, device=v.device)
    bk = triton.next_power_of_2(key_dim)
    if bk > 256:
        raise ValueError("key dimensions greater than 256 are unsupported")
    bv = min(value_block_size or 64, triton.next_power_of_2(value_dim))
    grid = (batch, h_v, triton.cdiv(value_dim, bv))
    _kda_target_verify_kernel[grid](
        A_log,
        dt_bias,
        q,
        k,
        v,
        a,
        b,
        initial_state_source,
        initial_state_indices,
        intermediate_states_buffer,
        intermediate_state_indices,
        out,
        scale,
        lower_bound if lower_bound is not None else 0.0,
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        a.stride(0),
        a.stride(1),
        a.stride(2),
        b.stride(0),
        b.stride(1),
        initial_state_source.stride(0),
        initial_state_source.stride(1),
        initial_state_source.stride(2),
        initial_state_source.stride(3),
        intermediate_states_buffer.stride(0),
        intermediate_states_buffer.stride(1),
        intermediate_states_buffer.stride(2),
        intermediate_states_buffer.stride(3),
        intermediate_states_buffer.stride(4),
        H_Q=h_q,
        H_K=h_k,
        H_V=h_v,
        K=key_dim,
        V=value_dim,
        STEPS=cache_steps,
        BK=bk,
        BV=bv,
        GATES_ARE_PREACTIVATED=gates_are_preactivated,
        USE_LOWER_BOUND=lower_bound is not None,
        PRECOMPUTE_RAW_GATES=precompute_raw_gates,
        BT=triton.next_power_of_2(cache_steps) if precompute_raw_gates else 1,
        num_warps=1,
        num_stages=3,
        multibuffer=False,
    )
    return out
