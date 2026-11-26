# Adapt from https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/fla/layernorm_gated.py to support npu
# Adapt from https://github.com/fla-org/flash-linear-attention/blob/main/fla/modules/layernorm_gated.py
# Copyright (c) 2024, Tri Dao.
# Based on the Triton LayerNorm tutorial: https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html

import torch
import triton
import triton.language as tl
from sgl_kernel_npu.implementation.triton.utils.triton_utils import (
    get_device_properties,
)

# Row tile budget in elements of the FP32 activation tile held in UB.
# The kernel keeps the tile (FP32), its square/deviation, the broadcast
# statistics and the compiler-inserted reduce/broadcast temp buffers alive at
# the same time, so the budget is much smaller than the 192KB UB. 8192 elements
# (32KB FP32 per tile) is the largest tile that still compiles with two
# reduction temp buffers and leaves room for double buffering; larger tiles
# either overflow UB (ascend910_9362 needs 3153920 bits for 256x128) or lose
# more time to non-overlapped vector work than they save in loop overhead.
_TILE_ELEMS = 8192
# Cap on rows per tile: with a small feature size the element budget would
# allow very tall tiles, but tall tiles grow the per-row reduce temp buffer and
# the row-broadcast traffic without improving DMA efficiency.
_MAX_BLOCK_M = 64


def _choose_block_m(block_n: int, rows: int) -> int:
    """Pick the row tile size for a BLOCK_N-wide FP32 tile."""
    block_m = max(1, min(_MAX_BLOCK_M, _TILE_ELEMS // block_n))
    return max(1, min(block_m, triton.next_power_of_2(rows)))


@triton.heuristics({"HAS_BIAS": lambda args: args["B"] is not None})
@triton.heuristics({"HAS_Z": lambda args: args["Z"] is not None})
@triton.jit
def _layer_norm_fwd_1pass_kernel_npu(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    B,  # pointer to the biases
    Z,  # pointer to the other branch
    Mean,  # pointer to the mean
    Rstd,  # pointer to the 1/std
    stride_x_row,  # how much to increase the pointer when moving by 1 row
    stride_y_row,
    stride_z_row,
    M,  # number of rows in X
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_Z: tl.constexpr,
    NORM_BEFORE_GATE: tl.constexpr,
    IS_RMS_NORM: tl.constexpr,
):
    # One program owns a fixed grid-stride slice of the row blocks, so the grid
    # only has to be as large as the vector core count.
    pid_m = tl.program_id(0)
    group = tl.program_id(1)
    X += group * N
    Y += group * N
    if HAS_Z:
        Z += group * N
    W += group * N
    if HAS_BIAS:
        B += group * N
    if not IS_RMS_NORM:
        Mean += group * M
    Rstd += group * M

    # Load weight once (broadcasted over rows)
    cols = tl.arange(0, BLOCK_N)
    col_mask = cols < N
    w = tl.load(W + cols, mask=col_mask).to(tl.float32)
    if HAS_BIAS:
        b = tl.load(B + cols, mask=col_mask).to(tl.float32)

    n_row_blocks = tl.cdiv(M, BLOCK_M)
    for row_block in tl.range(pid_m, n_row_blocks, tl.num_programs(0)):
        # Compute row indices for this program
        rows = row_block * BLOCK_M + tl.arange(0, BLOCK_M)
        row_mask = rows < M

        # Load X: shape [BLOCK_M, BLOCK_N]
        x_ptrs = X + rows[:, None] * stride_x_row + cols[None, :]
        x = tl.load(x_ptrs, mask=row_mask[:, None] & col_mask[None, :]).to(tl.float32)

        # Load Z if needed
        if HAS_Z:
            z_ptrs = Z + rows[:, None] * stride_z_row + cols[None, :]
            z = tl.load(z_ptrs, mask=row_mask[:, None] & col_mask[None, :]).to(
                tl.float32
            )
            if not NORM_BEFORE_GATE:
                x *= z * tl.sigmoid(z)

        # Compute statistics per row
        if not IS_RMS_NORM:
            mean = tl.sum(x, axis=1) / N  # [BLOCK_M]
            xbar = tl.where(col_mask[None, :], x - mean[:, None], 0.0)
            var = tl.sum(xbar * xbar, axis=1) / N
            tl.store(Mean + rows, mean, mask=row_mask)
        else:
            xbar = tl.where(col_mask[None, :], x, 0.0)
            var = tl.sum(xbar * xbar, axis=1) / N

        rstd = 1.0 / tl.sqrt(var + eps)  # [BLOCK_M]
        tl.store(Rstd + rows, rstd, mask=row_mask)

        # Normalize
        if not IS_RMS_NORM:
            x_hat = (x - mean[:, None]) * rstd[:, None]
        else:
            x_hat = x * rstd[:, None]

        y = x_hat * w[None, :]
        if HAS_BIAS:
            y += b[None, :]

        # Post-gate
        if HAS_Z and NORM_BEFORE_GATE:
            y *= z * tl.sigmoid(z)

        # Store output
        y_ptrs = Y + rows[:, None] * stride_y_row + cols[None, :]
        tl.store(y_ptrs, y, mask=row_mask[:, None] & col_mask[None, :])


def layer_norm_fwd(
    x,
    weight,
    bias,
    eps,
    z=None,
    out=None,
    group_size=None,
    norm_before_gate=True,
    is_rms_norm=False,
    activation=None,
):
    M, N = x.shape
    if group_size is None:
        group_size = N
    ngroups = N // group_size

    # allocate output
    if out is None:
        out = torch.empty_like(x)
    mean = (
        torch.empty((ngroups * M,), dtype=torch.float32, device=x.device)
        if not is_rms_norm
        else None
    )
    rstd = torch.empty((ngroups * M,), dtype=torch.float32, device=x.device)
    if M == 0:
        return out, mean, rstd

    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(group_size))

    # BLOCK_M is derived from the FP32 UB budget, not from the core count: the
    # tile size that keeps the vector core busy is independent of how many
    # cores run it (28 on 950PR, 40 on ascend910_9362, ...).
    BLOCK_M = _choose_block_m(BLOCK_N, M)
    _, num_vectorcore = get_device_properties()
    # Keep the grid to a single wave so every core gets the same number of row
    # blocks; the in-kernel grid-stride loop walks the rest.
    num_blocks_m = min(triton.cdiv(M, BLOCK_M), max(1, num_vectorcore // ngroups))
    # Double buffering only pays off when a core runs the loop more than once;
    # for the small-M (decode) shapes the prologue/epilogue of the pipeline is
    # pure overhead.
    multibuffer = triton.cdiv(M, BLOCK_M) >= 2 * num_blocks_m

    # Now grid is (num blocks over M, num groups)
    grid = (num_blocks_m, ngroups)
    _layer_norm_fwd_1pass_kernel_npu[grid](
        x,
        out,
        weight,
        bias,
        z,
        mean,
        rstd,
        x.stride(0),
        out.stride(0),
        z.stride(0) if z is not None else 0,
        M,
        group_size,
        eps,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        NORM_BEFORE_GATE=norm_before_gate,
        IS_RMS_NORM=is_rms_norm,
        # Overlap the next tile's DMA with the current tile's statistics and
        # epilogue (~10% at M=254208 on ascend910_9362).
        multibuffer=multibuffer,
    )
    return out, mean, rstd
