from functools import lru_cache

import torch
import triton
import triton.language as tl


@lru_cache(maxsize=None)
def _vector_core_count(device_index):
    properties = triton.runtime.driver.active.utils.get_device_properties(device_index)
    return max(1, int(properties["num_vectorcore"]))


def _layer_partitions(tensor, num_requests, num_layers):
    # Amortize request-index loads over layers without oversubscribing AIVs.
    cores = _vector_core_count(tensor.device.index)
    return min(num_layers, max(1, cores // num_requests))


@triton.jit
def _scatter_kda_conv_snapshot_kernel(
    dst_ptr,
    src_ptr,
    dst_indices_ptr,
    src_indices_ptr,
    step_indices_ptr,
    dst_layer_stride,
    dst_req_stride,
    src_layer_stride,
    src_req_stride,
    src_step_stride,
    NUM_LAYERS: tl.constexpr,
    LAYER_PARTITIONS: tl.constexpr,
    SRC_REQ_SIZE: tl.constexpr,
    SRC_STEP_SIZE: tl.constexpr,
    DST_REQ_SIZE: tl.constexpr,
    TAIL_NUMEL: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Copy one accepted contiguous conv snapshot per request and layer."""
    pid = tl.program_id(0)
    layer_partition = pid % LAYER_PARTITIONS
    req_idx = pid // LAYER_PARTITIONS
    dst_idx = tl.load(dst_indices_ptr + req_idx).to(tl.int64)
    src_idx = tl.load(src_indices_ptr + req_idx).to(tl.int64)
    step_idx = tl.load(step_indices_ptr + req_idx).to(tl.int64)
    valid = (
        (dst_idx >= 0)
        & (dst_idx < DST_REQ_SIZE)
        & (src_idx >= 0)
        & (src_idx < SRC_REQ_SIZE)
        & (step_idx >= 0)
        & (step_idx < SRC_STEP_SIZE)
    )
    if not valid:
        return
    safe_dst_idx = tl.maximum(dst_idx, 0)
    safe_src_idx = tl.maximum(src_idx, 0)
    safe_step_idx = tl.maximum(step_idx, 0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = valid & (offsets < TAIL_NUMEL)
    for layer_idx in range(layer_partition, NUM_LAYERS, LAYER_PARTITIONS):
        src_offsets = (
            layer_idx * src_layer_stride
            + safe_src_idx * src_req_stride
            + safe_step_idx * src_step_stride
            + offsets
        )
        dst_offsets = (
            layer_idx * dst_layer_stride + safe_dst_idx * dst_req_stride + offsets
        )
        value = tl.load(src_ptr + src_offsets, mask=mask, other=0.0)
        tl.store(dst_ptr + dst_offsets, value, mask=mask)


@triton.jit
def _commit_kda_extended_conv_state_kernel(
    conv_states_ptr,
    dst_indices_ptr,
    src_indices_ptr,
    step_indices_ptr,
    layer_stride,
    req_stride,
    window_stride,
    NUM_LAYERS: tl.constexpr,
    LAYER_PARTITIONS: tl.constexpr,
    POOL_SIZE: tl.constexpr,
    DRAFT_TOKEN_NUM: tl.constexpr,
    SELF_COMMIT: tl.constexpr,
    TAIL_NUMEL: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Commit one accepted window from #35021's extended conv state."""
    pid = tl.program_id(0)
    layer_partition = pid % LAYER_PARTITIONS
    req_idx = pid // LAYER_PARTITIONS
    step_idx = tl.load(step_indices_ptr + req_idx).to(tl.int64)
    if step_idx < 0 or step_idx >= DRAFT_TOKEN_NUM:
        return
    if SELF_COMMIT:
        if step_idx == DRAFT_TOKEN_NUM - 1:
            return
    dst_idx = tl.load(dst_indices_ptr + req_idx).to(tl.int64)
    if SELF_COMMIT:
        src_idx = dst_idx
    else:
        src_idx = tl.load(src_indices_ptr + req_idx).to(tl.int64)
    valid = (
        (dst_idx >= 0)
        & (dst_idx < POOL_SIZE)
        & (src_idx >= 0)
        & (src_idx < POOL_SIZE)
        & (step_idx >= 0)
        & (step_idx < DRAFT_TOKEN_NUM)
    )
    if not valid:
        return
    if not SELF_COMMIT:
        if (dst_idx == src_idx) & (step_idx == DRAFT_TOKEN_NUM - 1):
            return
    safe_dst_idx = tl.maximum(dst_idx, 0)
    safe_src_idx = tl.maximum(src_idx, 0)
    safe_step_idx = tl.maximum(step_idx, 0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = valid & (offsets < TAIL_NUMEL)
    for layer_idx in range(layer_partition, NUM_LAYERS, LAYER_PARTITIONS):
        src_offsets = (
            layer_idx * layer_stride
            + safe_src_idx * req_stride
            + safe_step_idx * window_stride
            + offsets
        )
        dst_offsets = (
            layer_idx * layer_stride
            + safe_dst_idx * req_stride
            + (DRAFT_TOKEN_NUM - 1) * window_stride
            + offsets
        )
        value = tl.load(conv_states_ptr + src_offsets, mask=mask, other=0.0)
        tl.store(conv_states_ptr + dst_offsets, value, mask=mask)


def scatter_kda_conv_snapshot(
    dst: torch.Tensor,
    src: torch.Tensor,
    dst_indices: torch.Tensor,
    src_indices: torch.Tensor,
    step_indices: torch.Tensor,
) -> bool:
    """Fast path for KDA's contiguous NPU conv-snapshot layout.

    Copy each accepted snapshot in one tile and reuse request indices while
    traversing layers. Split layers only when requests alone do not occupy
    the vector cores. Unsupported layouts return ``False`` so callers can
    use the generic stride-aware fallback.
    """
    if (
        dst.ndim != 4
        or src.ndim != 5
        or not dst.is_contiguous()
        or not src.is_contiguous()
        or dst.shape[0] != src.shape[0]
        or dst.shape[2:] != src.shape[3:]
    ):
        return False
    num_requests = int(step_indices.shape[0])
    if num_requests == 0:
        return True
    if (
        dst_indices.ndim != 1
        or src_indices.ndim != 1
        or step_indices.ndim != 1
        or dst_indices.shape[0] != num_requests
        or src_indices.shape[0] != num_requests
        or any(
            value.dtype != torch.int32
            for value in (dst_indices, src_indices, step_indices)
        )
    ):
        return False
    tail_numel = int(dst.shape[2] * dst.shape[3])
    block_size = triton.next_power_of_2(tail_numel)
    if block_size > 8192:
        return False
    num_layers = int(dst.shape[0])
    layer_partitions = _layer_partitions(dst, num_requests, num_layers)
    _scatter_kda_conv_snapshot_kernel[(num_requests * layer_partitions,)](
        dst,
        src,
        dst_indices,
        src_indices,
        step_indices,
        dst.stride(0),
        dst.stride(1),
        src.stride(0),
        src.stride(1),
        src.stride(2),
        NUM_LAYERS=num_layers,
        LAYER_PARTITIONS=layer_partitions,
        SRC_REQ_SIZE=src.shape[1],
        SRC_STEP_SIZE=src.shape[2],
        DST_REQ_SIZE=dst.shape[1],
        TAIL_NUMEL=tail_numel,
        BLOCK_SIZE=block_size,
    )
    return True


def commit_kda_extended_conv_state(
    conv_states: torch.Tensor,
    dst_indices: torch.Tensor,
    src_indices: torch.Tensor,
    step_indices: torch.Tensor,
    draft_token_num: int,
) -> bool:
    """Commit accepted KDA conv windows from #35021's extended state layout.

    ``conv_states`` is ``[layers, pool, base_window + draft_tokens - 1, channels]``.
    Each accepted state is an overlapping base-window slice beginning at its
    step index. The kernel copies that slice directly to the persistent tail,
    preserving #35021's CANN verify layout without the serial rollback loop.
    A fully accepted primary slot already contains that tail; tracking a
    different destination must still copy it.
    """
    if (
        conv_states.ndim != 4
        or not conv_states.is_contiguous()
        or draft_token_num <= 0
        or conv_states.shape[2] < draft_token_num
    ):
        return False
    num_requests = int(step_indices.shape[0])
    if num_requests == 0:
        return True
    if (
        dst_indices.ndim != 1
        or src_indices.ndim != 1
        or step_indices.ndim != 1
        or dst_indices.shape[0] != num_requests
        or src_indices.shape[0] != num_requests
        or any(
            value.dtype != torch.int32
            for value in (dst_indices, src_indices, step_indices)
        )
    ):
        return False
    base_window_size = int(conv_states.shape[2] - (draft_token_num - 1))
    num_dims = int(conv_states.shape[3])
    tail_numel = base_window_size * num_dims
    block_size = triton.next_power_of_2(tail_numel)
    if base_window_size <= 0 or block_size > 8192:
        return False
    num_layers = int(conv_states.shape[0])
    self_commit = dst_indices is src_indices
    # Primary requests may all be padding or fully accepted. Keep one program
    # per request so those cases do not launch idle layer partitions.
    layer_partitions = (
        1 if self_commit else _layer_partitions(conv_states, num_requests, num_layers)
    )
    _commit_kda_extended_conv_state_kernel[(num_requests * layer_partitions,)](
        conv_states,
        dst_indices,
        src_indices,
        step_indices,
        conv_states.stride(0),
        conv_states.stride(1),
        conv_states.stride(2),
        NUM_LAYERS=num_layers,
        LAYER_PARTITIONS=layer_partitions,
        POOL_SIZE=conv_states.shape[1],
        DRAFT_TOKEN_NUM=draft_token_num,
        SELF_COMMIT=self_commit,
        TAIL_NUMEL=tail_numel,
        BLOCK_SIZE=block_size,
    )
    return True


@triton.jit
def _move_kda_temporal_snapshot_kernel(
    dst_ptr,
    src_ptr,
    dst_indices_ptr,
    src_indices_ptr,
    step_indices_ptr,
    dst_layer_stride,
    dst_req_stride,
    dst_head_stride,
    dst_v_stride,
    dst_k_stride,
    src_layer_stride,
    src_req_stride,
    src_step_stride,
    src_head_stride,
    src_v_stride,
    src_k_stride,
    src_req_size,
    src_step_size,
    dst_req_size,
    head_dim,
    dim_v,
    dim_k,
    NUM_LAYERS: tl.constexpr,
    LAYER_PARTITIONS: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Copy one KDA temporal snapshot with layer-level parallelism."""
    pid = tl.program_id(0)
    layer_partition = pid % LAYER_PARTITIONS
    req_idx = pid // LAYER_PARTITIONS
    dst_idx = tl.load(dst_indices_ptr + req_idx).to(tl.int64)
    src_idx = tl.load(src_indices_ptr + req_idx).to(tl.int64)
    step_idx = tl.load(step_indices_ptr + req_idx).to(tl.int64)
    valid = (
        (dst_idx >= 0)
        & (dst_idx < dst_req_size)
        & (src_idx >= 0)
        & (src_idx < src_req_size)
        & (step_idx >= 0)
        & (step_idx < src_step_size)
    )
    if not valid:
        return

    v_offsets = tl.arange(0, BLOCK_V)
    k_offsets = tl.arange(0, BLOCK_K)
    for layer_idx in range(layer_partition, NUM_LAYERS, LAYER_PARTITIONS):
        for head_idx in range(head_dim):
            for v_start in range(0, dim_v, BLOCK_V):
                v_idx = v_start + v_offsets
                mask = (v_idx[:, None] < dim_v) & (k_offsets[None, :] < dim_k)
                src_offsets = (
                    layer_idx * src_layer_stride
                    + src_idx * src_req_stride
                    + step_idx * src_step_stride
                    + head_idx * src_head_stride
                    + v_idx[:, None] * src_v_stride
                    + k_offsets[None, :] * src_k_stride
                )
                dst_offsets = (
                    layer_idx * dst_layer_stride
                    + dst_idx * dst_req_stride
                    + head_idx * dst_head_stride
                    + v_idx[:, None] * dst_v_stride
                    + k_offsets[None, :] * dst_k_stride
                )
                value = tl.load(src_ptr + src_offsets, mask=mask, other=0.0)
                tl.store(dst_ptr + dst_offsets, value, mask=mask)


def move_kda_temporal_snapshot(
    dst: torch.Tensor,
    src: torch.Tensor,
    dst_indices: torch.Tensor,
    src_indices: torch.Tensor,
    step_indices: torch.Tensor,
) -> bool:
    """Fast path for Kimi-K3's production temporal-state layout.

    The generic mover assigns one program to a request, so batch size one
    serializes every KDA layer and head. Split layers over available vector
    cores while retaining loops over heads and 64-row transpose tiles. Only
    batch size one uses this path until full-shape batched measurements are
    available. Other requests/layouts retain the existing mover.
    """
    if (
        dst.ndim != 5
        or src.ndim != 6
        or dst.shape[0] != src.shape[0]
        or dst.shape[2:] != src.shape[3:]
        or dst.dtype != src.dtype
        or not src.is_contiguous()
    ):
        return False
    num_requests = int(step_indices.shape[0])
    if num_requests == 0:
        return True
    if (
        dst_indices.ndim != 1
        or src_indices.ndim != 1
        or step_indices.ndim != 1
        or dst_indices.shape[0] != num_requests
        or src_indices.shape[0] != num_requests
        or any(
            value.dtype != torch.int32
            for value in (dst_indices, src_indices, step_indices)
        )
    ):
        return False
    dim_k = int(src.shape[-1])
    block_k = triton.next_power_of_2(dim_k)
    if block_k > 128:
        return False
    num_layers = int(src.shape[0])
    layer_partitions = _layer_partitions(dst, num_requests, num_layers)
    if num_requests != 1 or layer_partitions == 1:
        # Batched temporal measurements did not justify replacing the mover.
        return False
    _move_kda_temporal_snapshot_kernel[(num_requests * layer_partitions,)](
        dst,
        src,
        dst_indices,
        src_indices,
        step_indices,
        dst.stride(0),
        dst.stride(1),
        dst.stride(2),
        dst.stride(3),
        dst.stride(4),
        src.stride(0),
        src.stride(1),
        src.stride(2),
        src.stride(3),
        src.stride(4),
        src.stride(5),
        src.shape[1],
        src.shape[2],
        dst.shape[1],
        src.shape[3],
        src.shape[4],
        dim_k,
        NUM_LAYERS=num_layers,
        LAYER_PARTITIONS=layer_partitions,
        BLOCK_V=64,
        BLOCK_K=block_k,
    )
    return True


__all__ = [
    "commit_kda_extended_conv_state",
    "move_kda_temporal_snapshot",
    "scatter_kda_conv_snapshot",
]
