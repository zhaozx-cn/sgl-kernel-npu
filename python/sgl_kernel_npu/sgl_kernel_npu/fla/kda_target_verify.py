from typing import Optional

import torch

from sgl_kernel_npu.fla.utils import input_guard


@input_guard
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
    safe_gate: bool = False,
    lower_bound: float = -5.0,
) -> torch.Tensor:
    """KDA fixed-width target verification backed by the AscendC recurrent_kda kernel.

    The kernel applies the full gate contract internally
    (``exp(-exp(A_log) * softplus(a + dt_bias))`` when safe_gate=False,
    ``lower_bound * sigmoid(-exp(A_log) * (a + dt_bias))`` when safe_gate=True)
    and ``sigmoid(b)`` in-kernel, so callers must pass the *raw* (un-preactivated)
    gate ``a`` and beta ``b`` together with ``A_log`` and ``dt_bias``.

    The persistent and intermediate state layout is the Ascend KDA layout
    ``[..., H_v, V, K]``.
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
    # ``a`` may carry a leading singleton [1, T, H_k, K] or be [T, H_k, K].
    # ``b`` may carry a leading singleton [1, T, H_v] or be [T, H_v].
    # Normalize both to their stripped forms.
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
    if A_log.numel() != h_k or tuple(dt_bias.shape) != (h_k, key_dim):
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

    tensors = [
        A_log, dt_bias, q, k, v, a, b,
        initial_state_source, initial_state_indices,
        intermediate_states_buffer, intermediate_state_indices,
    ]
    if any(t.device != q.device for t in tensors):
        raise ValueError("all tensors must be on the same device")
    initial_state_indices = initial_state_indices.contiguous()
    intermediate_state_indices = intermediate_state_indices.contiguous()
    if initial_state_source.dtype != intermediate_states_buffer.dtype:
        raise ValueError("persistent and intermediate state dtypes must match")
    if initial_state_source.dtype != torch.bfloat16:
        raise ValueError("recurrent_kda kernel currently requires bfloat16 state")
    if initial_state_indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("initial_state_indices must be int32 or int64")
    if intermediate_state_indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("intermediate_state_indices must be int32 or int64")

    if scale is None:
        scale = key_dim**-0.5
    if scale <= 0:
        raise ValueError("scale must be positive")

    # Reshape intermediate buffer from [scratch, steps, H_v, V, K] to
    # [scratch*steps, H_v, V, K] so the kernel treats each (batch, step) pair
    # as an independent state slot. The buffer must be contiguous so that
    # in-place state writes are visible to the caller through the original
    # intermediate_states_buffer reference.
    if not intermediate_states_buffer.is_contiguous():
        raise ValueError("intermediate_states_buffer must be contiguous")
    state_pool = intermediate_states_buffer.view(
        -1, h_v, value_dim, key_dim
    )

    # Pre-copy initial states from the persistent pool into the intermediate
    # buffer at slot 0 of each batch's section. The recurrent_kda kernel reads
    # the initial state from ssm_state_indices[batch, 0] and writes per-step
    # states to ssm_state_indices[batch, step].
    init_indices_flat = (
        intermediate_state_indices[:batch].to(torch.int64) * cache_steps
    )
    src_states = initial_state_source[
        initial_state_indices[:batch].to(torch.int64)
    ]
    state_pool.index_copy_(0, init_indices_flat, src_states)

    # Build cu_seqlens: [0, steps, 2*steps, ..., batch*steps]
    cu_seqlens = torch.arange(
        0,
        batch * cache_steps + 1,
        step=cache_steps,
        dtype=torch.int32,
        device=q.device,
    )

    # Build 2D ssm_state_indices [batch, steps] where element [i, j] maps to
    # the flattened intermediate buffer slot for batch i, step j.
    base_slots = (
        intermediate_state_indices[:batch]
        .to(torch.int32)
        .unsqueeze(1)
        .expand(batch, cache_steps)
    )
    step_offsets = torch.arange(
        cache_steps, dtype=torch.int32, device=q.device
    ).unsqueeze(0)
    ssm_state_indices = base_slots * cache_steps + step_offsets

    # Expand gate from H_k to H_v heads (each H_k head serves H_v/H_k value heads).
    if h_v != h_k:
        repeat_factor = h_v // h_k
        gate_expanded = a.repeat_interleave(repeat_factor, dim=1)
    else:
        gate_expanded = a

    # Reshape to BSND [1, tokens, H_v, K] for the kernel.
    gate_bsnd = gate_expanded.unsqueeze(0).contiguous()
    beta_bsnd = b.unsqueeze(0).contiguous()
    q_bsnd = q.contiguous()
    k_bsnd = k.contiguous()
    v_bsnd = v.contiguous()

    # The recurrent_kda kernel computes the full gate contract internally:
    #   exp(-exp(A_log) * softplus(a + dt_bias))   (safe_gate=False)
    #   lower_bound * sigmoid(-exp(A_log) * (a + dt_bias))  (safe_gate=True)
    # and sigmoid(b), so we always pass the raw gate and beta.
    out = torch.ops.npu.recurrent_kda(
        q_bsnd,
        k_bsnd,
        v_bsnd,
        gate_bsnd,
        beta_bsnd,
        state_pool,
        cu_seqlens,
        ssm_state_indices,
        a_log=A_log,
        dt_bias=dt_bias,
        scale=scale,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        allow_neg_eigval=False,
        safe_gate=safe_gate,
        lower_bound=lower_bound,
        state_v_first=True,
    )
    return out
