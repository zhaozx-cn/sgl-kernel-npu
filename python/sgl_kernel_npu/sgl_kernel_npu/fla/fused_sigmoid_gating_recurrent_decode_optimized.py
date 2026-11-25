from typing import Optional

import torch
import triton
import triton.language as tl

from sgl_kernel_npu.utils.triton_utils import get_device_properties


def _maybe_contiguous(x):
    """Return ``x`` unchanged when already contiguous, else a contiguous copy."""
    return x if x.is_contiguous() else x.contiguous()


# Pointer arguments whose alignment never needs to be part of the dispatch key.
# Decode shapes come from the model config, so ``q`` and friends move between
# allocations with different alignments at run time; specializing on it would
# only add cache misses on the host side.
_PTR_ARGS = [
    "A_log",
    "a",
    "dt_bias",
    "q",
    "k",
    "v",
    "b",
    "o",
    "h0_source",
    "h0_indices",
    "cu_seqlens",
]


@triton.jit(
    do_not_specialize=["T", "N", "NHV"],
    do_not_specialize_on_alignment=_PTR_ARGS,
)
def _fused_sigmoid_gating_delta_rule_update_decode_kernel(
    A_log,
    a,
    dt_bias,
    softplus_beta,
    softplus_threshold,
    q,
    k,
    v,
    b,
    o,
    h0_source,
    h0_indices,
    cu_seqlens,
    scale,
    T,
    N,
    NHV,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    EVEN_K: tl.constexpr,
    EVEN_V: tl.constexpr,
):
    """
    Decode-optimized fused gating + recurrent delta rule update.

    One program instance owns one ``(sequence, value head)`` group: the gating
    scalars and the normalized q/k vectors are produced once and reused by every
    value tile of that head, and each value tile is a single read-modify-write of
    a contiguous ``[BV, BK]`` state block.
    """
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)

    o_k = tl.arange(0, BK)
    o_base_v = tl.arange(0, BV)
    mask_k = o_k < K
    hv_per_h = HV // H

    total_tiles = N * NHV

    for g_idx in tl.range(pid, total_tiles, num_programs):
        i_n = g_idx // NHV
        i_nhv = g_idx % NHV
        i_hv = i_nhv
        i_h = i_hv // hv_per_h

        if IS_VARLEN:
            bos = tl.load(cu_seqlens + i_n).to(tl.int32)
            eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
            t_len = eos - bos
        else:
            bos = i_n * T
            t_len = T

        # Gating: computed once per (sequence, value-head) and reused by every
        # value tile and time step of that head.
        b_A_log = tl.load(A_log + i_hv).to(tl.float32)
        b_a = tl.load(a + bos * HV + i_hv).to(tl.float32)
        b_dt_bias = tl.load(dt_bias + i_hv).to(tl.float32)
        b_b = tl.load(b + bos * HV + i_hv).to(tl.float32)

        x = b_a + b_dt_bias
        beta_x = softplus_beta * x
        softplus_x = tl.where(
            beta_x <= softplus_threshold,
            (1.0 / softplus_beta) * tl.log(1.0 + tl.exp(beta_x)),
            x,
        )
        b_g = -tl.exp(b_A_log) * softplus_x
        b_decay = tl.exp(b_g)
        b_beta = 1.0 / (1.0 + tl.exp(-b_b))

        # q/k are shared across the value dimension; set pointers once.
        p_q = q + (bos * H + i_h) * K + o_k
        p_k = k + (bos * H + i_h) * K + o_k

        if USE_INITIAL_STATE:
            idx = tl.load(h0_indices + i_n)

        for i in tl.range(0, t_len):
            if EVEN_K:
                b_q = tl.load(p_q + i * H * K).to(tl.float32)
                b_k = tl.load(p_k + i * H * K).to(tl.float32)
            else:
                b_q = tl.load(p_q + i * H * K, mask=mask_k).to(tl.float32)
                b_k = tl.load(p_k + i * H * K, mask=mask_k).to(tl.float32)

            if USE_QK_L2NORM_IN_KERNEL:
                b_q = b_q / (tl.sqrt(tl.sum(b_q * b_q)) + 1e-6)
                b_k = b_k / (tl.sqrt(tl.sum(b_k * b_k)) + 1e-6)

            b_q = b_q * scale

            # Value tiles.  ``NV`` is a constexpr so the tile loop is fully
            # unrolled at compile time.
            for i_v in tl.static_range(0, NV):
                o_v = i_v * BV + o_base_v
                mask_v = o_v < V

                p_v = v + (bos * HV + i_hv) * V + o_v
                p_o = o + (bos * HV + i_hv) * V + o_v

                if USE_INITIAL_STATE:
                    p_h0 = (
                        h0_source
                        + idx.to(tl.int64) * HV * K * V
                        + i_hv * K * V
                        + o_v[:, None] * K
                        + o_k[None, :]
                    )

                if EVEN_V:
                    b_v = tl.load(p_v + i * HV * V).to(tl.float32)
                else:
                    b_v = tl.load(p_v + i * HV * V, mask=mask_v).to(tl.float32)

                if USE_INITIAL_STATE:
                    if idx >= 0:
                        if EVEN_K and EVEN_V:
                            b_h = tl.load(p_h0 + i * HV * K * V).to(tl.float32)
                        else:
                            b_h = tl.load(
                                p_h0 + i * HV * K * V,
                                mask=mask_v[:, None] & mask_k[None, :],
                            ).to(tl.float32)
                    else:
                        b_h = tl.zeros([BV, BK], dtype=tl.float32)
                else:
                    b_h = tl.zeros([BV, BK], dtype=tl.float32)

                # Recurrent delta rule update. b_h is [BV, BK] (V row, K col).
                b_h *= b_decay
                b_v -= tl.sum(b_h * b_k[None, :], 1)
                b_v *= b_beta
                b_h += b_k[None, :] * b_v[:, None]
                b_o = tl.sum(b_h * b_q[None, :], 1)

                if USE_INITIAL_STATE:
                    if idx >= 0:
                        b_h_store = b_h.to(h0_source.dtype.element_ty)
                        if EVEN_K and EVEN_V:
                            tl.store(p_h0 + i * HV * K * V, b_h_store)
                        else:
                            tl.store(
                                p_h0 + i * HV * K * V,
                                b_h_store,
                                mask=mask_v[:, None] & mask_k[None, :],
                            )

                if EVEN_V:
                    tl.store(p_o + i * HV * V, b_o.to(o.dtype.element_ty))
                else:
                    tl.store(
                        p_o + i * HV * V, b_o.to(o.dtype.element_ty), mask=mask_v
                    )


# --------------------------------------------------------------------------- #
# Host wrapper
# --------------------------------------------------------------------------- #
def fused_sigmoid_gating_delta_rule_update_decode_npu(
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    softplus_beta: float,
    softplus_threshold: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    b: torch.Tensor,
    initial_state_source: torch.Tensor,
    initial_state_indices: torch.Tensor,
    scale: Optional[float] = None,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
):
    # Decode inputs are already contiguous, so the generic decorator is
    # skipped in favor of a cheap contiguous check.
    A_log = _maybe_contiguous(A_log)
    a = _maybe_contiguous(a)
    dt_bias = _maybe_contiguous(dt_bias)
    q = _maybe_contiguous(q)
    k = _maybe_contiguous(k)
    v = _maybe_contiguous(v)
    b = _maybe_contiguous(b)
    initial_state_source = _maybe_contiguous(initial_state_source)
    initial_state_indices = _maybe_contiguous(initial_state_indices)
    if cu_seqlens is not None:
        cu_seqlens = _maybe_contiguous(cu_seqlens)

    # Triton resolves the target device from the current device and needs the
    # raw stream of that device, so both are taken from the tensors rather than
    # from whatever device the caller happens to have left current.
    device = q.device.index
    if device is None:
        device = torch.npu.current_device()

    B, T, H, K = q.shape
    HV = v.shape[2]
    V = v.shape[-1]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1

    # Single-token decode only.  In varlen mode q.shape[1] is the total token
    # count across the batch and must equal the sequence count.
    if cu_seqlens is None:
        if T != 1:
            raise ValueError(
                f"decode kernel only supports T=1 per sequence, got T={T}"
            )
    elif T != N:
        raise ValueError(
            "decode kernel only supports 1 token per sequence, got "
            f"{T} tokens across {N} sequences"
        )

    # ``triton.next_power_of_2`` / ``triton.cdiv`` are ``ConstexprFunction``
    # objects whose host-side call costs ~1.3 us each; the plain integer forms
    # below are equivalent and effectively free.
    BK = 1 << (K - 1).bit_length()
    BV = min(1 << (V - 1).bit_length(), 64)
    NK = (K + BK - 1) // BK
    NV = (V + BV - 1) // BV
    assert NK == 1, "NK > 1 is not supported in the decode-optimized kernel"

    if scale is None:
        scale = K**-0.5
    else:
        assert scale > 0, "scale must be positive"

    o = q.new_empty(B, T, HV, V)

    # 1-D grid sized to the AIV vector core count; one resident instance per
    # core is enough because the loop trip count is already small.
    num_aicore, num_vectorcore = get_device_properties()
    OVERSUB = 1
    num_programs = min(N * HV, num_vectorcore * OVERSUB)
    num_programs = max(1, num_programs)
    grid = (num_programs,)

    use_initial_state = initial_state_source is not None
    is_varlen = cu_seqlens is not None

    # Unused pointer parameters are replaced by a live tensor because the kernel
    # does not accept ``None`` for pointer arguments; the branches that would
    # read them are removed by the constexpr guards.
    h0_ptr = initial_state_source if use_initial_state else q
    idx_ptr = initial_state_indices if use_initial_state else q
    seq_ptr = cu_seqlens if is_varlen else q
    even_k = BK == K
    even_v = (V % BV) == 0

    # Triton resolves the target device from the current device, so the tensors'
    # device is entered explicitly instead of assuming the caller already did.
    with torch.npu.device(device):
        _fused_sigmoid_gating_delta_rule_update_decode_kernel[grid](
            A_log,
            a,
            dt_bias,
            softplus_beta,
            softplus_threshold,
            q,
            k,
            v,
            b,
            o,
            h0_ptr,
            idx_ptr,
            seq_ptr,
            scale,
            T,
            N,
            HV,
            H,
            HV,
            K,
            V,
            BK,
            BV,
            NV,
            use_initial_state,
            is_varlen,
            use_qk_l2norm_in_kernel,
            even_k,
            even_v,
            num_warps=4,
            num_stages=1,
            multibuffer=False,
        )
    return o
