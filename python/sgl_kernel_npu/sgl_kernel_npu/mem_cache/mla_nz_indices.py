"""Physical NZ row indices for paged MLA latent and RoPE scatter writes."""

import torch
import triton
import triton.language as tl


@triton.jit
def _mla_nz_scatter_indices_kernel(
    locations,
    latent_indices,
    rope_indices,
    rows,
    location_stride,
    PAGE_SIZE: tl.constexpr,
    LATENT_TILES: tl.constexpr,
    ROPE_TILES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    k_row, k_tile = offsets // LATENT_TILES, offsets % LATENT_TILES
    k_loc = tl.load(locations + k_row * location_stride, k_row < rows, other=0)
    k_index = (
        k_loc // PAGE_SIZE * LATENT_TILES + k_tile
    ) * PAGE_SIZE + k_loc % PAGE_SIZE
    tl.store(latent_indices + offsets, k_index, k_row < rows)
    r_row, r_tile = offsets // ROPE_TILES, offsets % ROPE_TILES
    r_loc = tl.load(locations + r_row * location_stride, r_row < rows, other=0)
    r_index = (r_loc // PAGE_SIZE * ROPE_TILES + r_tile) * PAGE_SIZE + r_loc % PAGE_SIZE
    tl.store(rope_indices + offsets, r_index, r_row < rows)


def build_mla_nz_scatter_indices(locations, latent_dim, rope_dim, page_size):
    """Build both physical-row index arrays without staging intermediate tensors.

    Logical locations are nonnegative token slots. Physical indices must fit
    in the input integer dtype, matching the caller's scatter contract.
    Results are [tokens * (dim / 16), 1] in token/tile order. Location values
    remain live under graph replay, including non-contiguous input vectors.
    """
    if locations.ndim != 1 or locations.dtype not in (torch.int32, torch.int64):
        raise ValueError("locations must be a vector of int32 or int64 slot IDs")
    if page_size <= 0 or any(dim <= 0 or dim % 16 for dim in (latent_dim, rope_dim)):
        raise ValueError(
            "page_size must be positive and dimensions positive multiples of 16"
        )
    rows = locations.numel()
    k_tiles, r_tiles = latent_dim // 16, rope_dim // 16
    k = torch.empty((rows * k_tiles, 1), dtype=locations.dtype, device=locations.device)
    r = torch.empty((rows * r_tiles, 1), dtype=locations.dtype, device=locations.device)
    if rows:
        _mla_nz_scatter_indices_kernel[
            (triton.cdiv(rows * max(k_tiles, r_tiles), 256),)
        ](
            locations,
            k,
            r,
            rows,
            locations.stride(0),
            PAGE_SIZE=page_size,
            LATENT_TILES=k_tiles,
            ROPE_TILES=r_tiles,
            BLOCK=256,
            num_warps=4,
        )
    return k, r
