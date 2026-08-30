/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See the License for the full text of the License.
 */

/*!
 * \file chunk_kda_fwd.cpp
 * \brief host-side direct-launch implementation of chunk_kda_fwd
 *
 * Ports the aclnn host flow (aclnn_chunk_kda_fwd.cpp + chunk_kda_fwd_tiling.cpp)
 * into a single host function: layout normalization, input validation, tiling
 * computation and kernel launch.
 */

#include <algorithm>
#include <cstring>
#include <tuple>
#include <vector>

#include "acl/acl.h"

#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/DeviceUtils.h"
#include "torch_npu/csrc/framework/OpCommand.h"

#include "platform/soc_spec.h"
#include "tiling/platform/platform_ascendc.h"

#include "defines.h"
#include "torch_helper.h"
#include "aclrtlaunch_chunk_kda_fwd.h"

#include "../op_kernel/chunk_kda_fwd_tiling_data.h"
#include "arch35/chunk_kda_fwd_tiling_impl.h"

namespace sglang {
namespace npu_kernel {

namespace {

constexpr int64_t SYS_WORKSPACE_SIZE = 16777216;  // 16 MB reserved system workspace
constexpr int64_t KDA_DTYPE_FP16 = 0;
constexpr int64_t KDA_DTYPE_BF16 = 1;
constexpr int64_t KDA_DTYPE_FP32 = 2;

constexpr uint64_t KDA_ALIGN = 512;
constexpr uint64_t KDA_SOLVE_SCRATCH_SLOTS = 5;
constexpr uint64_t KDA_SOLVE_PIPELINE_DEPTH = 4;
constexpr uint64_t KDA_SCORE_QUEUE_SLOTS = 4;
constexpr uint64_t KDA_SCORE_SCRATCH_PLANES = 3;
constexpr uint64_t KDA_GDN_PIPELINE_DEPTH = 2;
constexpr int64_t MAX_KDA_K_DIM = 256;
constexpr int64_t MAX_KDA_HEAD_NUM = 128;
constexpr int64_t MAX_KDA_VARLEN_SEQUENCES = 1024;
constexpr int64_t MAX_KDA_CHUNK = 128;

enum class KdaFwdLayout { BSND, BNSD, TND, NTD };

uint64_t AlignWorkspace(uint64_t bytes)
{
    return (bytes + KDA_ALIGN - 1) / KDA_ALIGN * KDA_ALIGN;
}

uint64_t AllocateWorkspace(uint64_t &cursor, uint64_t bytes)
{
    const uint64_t offset = AlignWorkspace(cursor);
    cursor = offset + bytes;
    return offset;
}

bool ParseLayout(c10::string_view layout, KdaFwdLayout &parsed)
{
    if (layout == "BSND") {
        parsed = KdaFwdLayout::BSND;
    } else if (layout == "BNSD") {
        parsed = KdaFwdLayout::BNSD;
    } else if (layout == "TND") {
        parsed = KdaFwdLayout::TND;
    } else if (layout == "NTD") {
        parsed = KdaFwdLayout::NTD;
    } else {
        return false;
    }
    return true;
}

std::vector<int64_t> ReadHostInt64(const at::Tensor &tensor)
{
    at::Tensor host = tensor.device().type() == c10::DeviceType::PrivateUse1 ? tensor.cpu() : tensor;
    TORCH_CHECK(host.scalar_type() == at::kLong, "cu_seqlens must be int64");
    const int64_t *data = host.data_ptr<int64_t>();
    return std::vector<int64_t>(data, data + host.numel());
}

// Computes seqNum/totalChunks from cu_seqlens host values.
bool ResolveSequenceInfo(const std::vector<int64_t> &cuHost, int64_t seqlen, int64_t chunkSize,
                         int64_t batch, bool &isVarLen, int64_t &seqNum, int64_t &totalChunks)
{
    isVarLen = !cuHost.empty();
    seqNum = batch;
    totalChunks = (seqlen + chunkSize - 1) / chunkSize;
    if (!isVarLen) {
        return totalChunks > 0;
    }
    seqNum = static_cast<int64_t>(cuHost.size()) - 1;
    if (seqNum <= 0 || cuHost[0] != 0 || cuHost[seqNum] > seqlen) {
        return false;
    }
    totalChunks = 0;
    for (int64_t seq = 0; seq < seqNum; ++seq) {
        if (cuHost[seq] < 0 || cuHost[seq + 1] < cuHost[seq]) {
            return false;
        }
        totalChunks += (cuHost[seq + 1] - cuHost[seq] + chunkSize - 1) / chunkSize;
    }
    return totalChunks > 0;
}

// Ports Tiling4ChunkKdaFwd (op_host/chunk_kda_fwd_tiling.cpp): fills the tiling
// struct and returns the exact tail workspace size (bytes after the reserved
// system workspace).
uint64_t ComputeTilingData(int64_t batch, int64_t seqNum, int64_t qHeads, int64_t vHeads, int64_t seqlen,
                           int64_t kDim, int64_t vDim, int64_t chunkSize, int64_t totalChunks,
                           bool sequenceMajor, float scale, float lowerBound, bool hasInitialState,
                           bool isVarLen, bool safeGate, bool useGateInKernel, bool hasALog, bool hasDtBias,
                           const optiling::arch35::ChunkKdaFwdArch35Options &arch35Options, int64_t gateDataType,
                           int64_t qDataType, int64_t betaDataType, int64_t dataBytes, uint32_t blockDim,
                           KdaForward::ChunkKdaFwdTilingData &td)
{
    std::memset(&td, 0, sizeof(td));

    td.batch = batch;
    td.seqNum = seqNum;
    td.qHeadNum = qHeads;
    td.vHeadNum = vHeads;
    td.seqlen = seqlen;
    td.kHeadDim = kDim;
    td.vHeadDim = vDim;
    td.chunkSize = chunkSize;
    td.totalChunks = totalChunks;
    td.inputRank = 4;  // the kernel always consumes rank-4 normalized views
    td.scale = scale;
    td.lowerBound = lowerBound;
    td.hasInitialState = hasInitialState;
    td.isVarLen = isVarLen;
    td.safeGate = safeGate;
    td.inputSequenceMajor = sequenceMajor;
    td.useGateInKernel = useGateInKernel;
    td.hasALog = hasALog;
    td.hasDtBias = hasDtBias;
    td.computeGateInPrepare = arch35Options.computeGateInPrepare;
    td.fusePostWu = arch35Options.fusePostWu;
    td.fusePostWuIntoFwdH = arch35Options.fusePostWuIntoFwdH;
    td.useDenseFwdH = arch35Options.useDenseFwdH;
    td.gateDataType = gateDataType;
    td.qDataType = qDataType;
    td.betaDataType = betaDataType;
    td.gateUsedCoreNum = static_cast<int64_t>(blockDim) * 2;
    td.prepareUsedCoreNum = blockDim;
    td.postWuUsedCoreNum = blockDim;
    td.outputUsedCoreNum = blockDim;

    const uint64_t tokenHeads = static_cast<uint64_t>(batch) * vHeads * seqlen;
    const uint64_t kTensorBytes = tokenHeads * kDim * dataBytes;
    const uint64_t vTensorBytes = tokenHeads * vDim * dataBytes;
    const uint64_t gkBytes = tokenHeads * kDim * sizeof(float);
    const uint64_t stateElements = static_cast<uint64_t>(seqNum) * vHeads * kDim * vDim;
    const uint64_t hChunkCount =
        isVarLen ? static_cast<uint64_t>(totalChunks) : static_cast<uint64_t>(batch) * totalChunks;
    const uint64_t hBytes = hChunkCount * vHeads * kDim * vDim * dataBytes;

    uint64_t cursor = 0;
    td.gkStorageOffset = static_cast<int64_t>(AllocateWorkspace(cursor, gkBytes));
    td.finalStateStorageOffset = static_cast<int64_t>(AllocateWorkspace(cursor, stateElements * sizeof(float)));
    td.wStorageOffset = static_cast<int64_t>(AllocateWorkspace(cursor, kTensorBytes));
    td.uStorageOffset = static_cast<int64_t>(AllocateWorkspace(cursor, vTensorBytes));
    td.qgStorageOffset = static_cast<int64_t>(AllocateWorkspace(cursor, kTensorBytes));
    td.kgStorageOffset = static_cast<int64_t>(AllocateWorkspace(cursor, kTensorBytes));
    const uint64_t vNewStorageBytes =
        arch35Options.useDenseFwdH ?
            static_cast<uint64_t>(batch) * vHeads * chunkSize * vDim * dataBytes :
            vTensorBytes;
    td.vNewStorageOffset = static_cast<int64_t>(AllocateWorkspace(cursor, vNewStorageBytes));
    const uint64_t hStorageBytes =
        arch35Options.useDenseFwdH ?
            static_cast<uint64_t>(batch) * vHeads * kDim * vDim * dataBytes :
            hBytes;
    td.hStorageOffset = static_cast<int64_t>(AllocateWorkspace(cursor, hStorageBytes));
    td.qgScaledOffset = static_cast<int64_t>(AllocateWorkspace(cursor, kTensorBytes));

    const uint64_t matrixBytes = tokenHeads * chunkSize * sizeof(float);
    td.prepareAqkFp32Offset = static_cast<int64_t>(AllocateWorkspace(cursor, matrixBytes));
    td.prepareAkkFp32Offset = static_cast<int64_t>(AllocateWorkspace(cursor, matrixBytes));
    td.prepareScratchOffset = static_cast<int64_t>(AlignWorkspace(cursor));
    const uint64_t solveDepth = safeGate ? KDA_SOLVE_PIPELINE_DEPTH : 1;
    const uint64_t solveBytes = static_cast<uint64_t>(blockDim) * solveDepth * KDA_SOLVE_SCRATCH_SLOTS *
                                chunkSize * chunkSize * sizeof(float);
    const uint64_t scoreBytes = static_cast<uint64_t>(blockDim) * KDA_SCORE_QUEUE_SLOTS *
                                KDA_SCORE_SCRATCH_PLANES * chunkSize * kDim * dataBytes;
    cursor = static_cast<uint64_t>(td.prepareScratchOffset) + AlignWorkspace(solveBytes) + scoreBytes;

    td.postWuScratchOffset = static_cast<int64_t>(AlignWorkspace(cursor));
    if (!arch35Options.fusePostWu && !arch35Options.fusePostWuIntoFwdH) {
        cursor = static_cast<uint64_t>(td.postWuScratchOffset) + tokenHeads * kDim * sizeof(float);
    }

    td.fwdHWorkspaceBaseOffset = static_cast<int64_t>(AlignWorkspace(cursor));
    uint64_t fwdHCursor = 0;
    td.vWorkspaceOffset = static_cast<int64_t>(AllocateWorkspace(
        fwdHCursor, static_cast<uint64_t>(blockDim) * chunkSize * vDim * sizeof(float) * KDA_GDN_PIPELINE_DEPTH));
    td.vUpdateWorkspaceOffset = static_cast<int64_t>(AllocateWorkspace(
        fwdHCursor, static_cast<uint64_t>(blockDim) * chunkSize * vDim * sizeof(float) * KDA_GDN_PIPELINE_DEPTH));
    td.kDecayWorkspaceOffset = static_cast<int64_t>(AllocateWorkspace(
        fwdHCursor, static_cast<uint64_t>(blockDim) * chunkSize * kDim * sizeof(float) * KDA_GDN_PIPELINE_DEPTH));
    td.hWorkspaceOffset = static_cast<int64_t>(AllocateWorkspace(
        fwdHCursor, static_cast<uint64_t>(blockDim) * kDim * vDim * sizeof(float) * KDA_GDN_PIPELINE_DEPTH));
    const uint64_t tokenBatch = isVarLen ? static_cast<uint64_t>(seqNum) : 1;
    td.numSeqWorkspaceOffset =
        static_cast<int64_t>(AllocateWorkspace(fwdHCursor, (tokenBatch + 1) * sizeof(int64_t)));
    td.numChunksWorkspaceOffset =
        static_cast<int64_t>(AllocateWorkspace(fwdHCursor, (tokenBatch + 1) * sizeof(int64_t)));
    cursor = static_cast<uint64_t>(td.fwdHWorkspaceBaseOffset) + AlignWorkspace(fwdHCursor);

    td.outputScratchOffset = static_cast<int64_t>(AllocateWorkspace(cursor, 2 * tokenHeads * vDim * sizeof(float)));
    return AlignWorkspace(cursor);
}

}  // namespace

HOST_API std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
                    at::Tensor, at::Tensor, at::Tensor, at::Tensor>
chunk_kda_fwd(const at::Tensor &q, const at::Tensor &k, const at::Tensor &v, const at::Tensor &g,
              const at::Tensor &beta, const c10::optional<at::Tensor> &a_log,
              const c10::optional<at::Tensor> &dt_bias, const c10::optional<at::Tensor> &initial_state,
              const c10::optional<at::Tensor> &cu_seqlens, const c10::optional<at::Tensor> &chunk_indices,
              c10::string_view layout, double scale, int64_t chunk_size, bool safe_gate, double lower_bound,
              bool use_gate_in_kernel, bool state_v_first, bool output_final_state, bool output_gk,
              bool output_w, bool output_u, bool output_qg, bool output_kg, bool output_v_new, bool output_h)
{
    TORCH_CHECK(q.defined() && k.defined() && v.defined() && g.defined() && beta.defined(),
                "q/k/v/g/beta must be defined");

    KdaFwdLayout parsedLayout = KdaFwdLayout::BSND;
    TORCH_CHECK(ParseLayout(layout, parsedLayout), "layout must be one of BSND, BNSD, TND or NTD");
    TORCH_CHECK(chunk_size == 64 || chunk_size == MAX_KDA_CHUNK, "chunk_size must be 64 or 128");
    const float scaleValue = static_cast<float>(scale);
    const float lowerBoundValue = static_cast<float>(lower_bound);

    const bool sequenceMajor = parsedLayout == KdaFwdLayout::BSND;
    const bool isRank3 = parsedLayout == KdaFwdLayout::TND || parsedLayout == KdaFwdLayout::NTD;

    // ---------------- dtype checks ----------------
    TORCH_CHECK(q.scalar_type() == at::kBFloat16 && k.scalar_type() == at::kBFloat16 &&
                    v.scalar_type() == at::kBFloat16,
                "q/k/v must be bfloat16 (the direct-launch kernel is compiled for bfloat16 only)");
    TORCH_CHECK(g.scalar_type() == at::kFloat || g.scalar_type() == at::kBFloat16,
                "g must be float32 or bfloat16");
    TORCH_CHECK(beta.scalar_type() == at::kFloat || beta.scalar_type() == at::kBFloat16,
                "beta must be float32 or bfloat16");
    const int64_t qDataType = KDA_DTYPE_BF16;
    const int64_t gateDataType =
        g.scalar_type() == at::kFloat ? KDA_DTYPE_FP32 : KDA_DTYPE_BF16;
    const int64_t betaDataType = beta.scalar_type() == at::kFloat ? KDA_DTYPE_FP32 : KDA_DTYPE_BF16;

    // ---------------- layout normalization ----------------
    // The kernel consumes rank-4 views: BSND stays sequence-major, every other
    // layout is presented head-major (BNSD-style). beta must always be
    // head-major (B, HV, S). TND inputs are transposed on device, matching the
    // aclnn executor behaviour.
    at::Tensor qK = q.contiguous();
    at::Tensor kK = k.contiguous();
    at::Tensor vK = v.contiguous();
    at::Tensor gK = g.contiguous();
    at::Tensor betaK = beta.contiguous();
    if (parsedLayout == KdaFwdLayout::BSND) {
        betaK = betaK.transpose(1, 2).contiguous();
    } else if (parsedLayout == KdaFwdLayout::TND) {
        qK = qK.permute({1, 0, 2}).contiguous();
        kK = kK.permute({1, 0, 2}).contiguous();
        vK = vK.permute({1, 0, 2}).contiguous();
        gK = gK.permute({1, 0, 2}).contiguous();
        betaK = betaK.permute({1, 0}).contiguous();
    }

    TORCH_CHECK(qK.dim() == 4 && kK.dim() == 4 && vK.dim() == 4 && gK.dim() == 4 && betaK.dim() == 3,
                "normalized q/k/v/g must be rank-4 and beta rank-3");
    const int64_t batch = qK.size(0);
    const int64_t qHeads = qK.size(1);
    const int64_t seqlen = qK.size(2);
    const int64_t kDim = qK.size(3);
    const int64_t vHeads = vK.size(1);
    const int64_t vDim = vK.size(3);

    TORCH_CHECK(qK.sizes() == kK.sizes(), "q and k must have identical shape");
    TORCH_CHECK(vK.size(0) == batch && vK.size(2) == seqlen, "v must be (B, HV, S, V) after normalization");
    TORCH_CHECK(gK.size(0) == batch && gK.size(1) == vHeads && gK.size(2) == seqlen && gK.size(3) == kDim,
                "g must be (B, HV, S, K) after normalization");
    TORCH_CHECK(betaK.size(0) == batch && betaK.size(1) == vHeads && betaK.size(2) == seqlen,
                "beta must be (B, HV, S) after normalization");
    if (isRank3) {
        TORCH_CHECK(batch == 1, "rank-3 layouts expect a single (virtual) batch");
    }

    TORCH_CHECK(qHeads > 0 && vHeads >= qHeads && vHeads % qHeads == 0,
                "H and HV must be positive, HV must be greater than or equal to H, and HV must be divisible by H");
    TORCH_CHECK(qHeads <= MAX_KDA_HEAD_NUM && vHeads <= MAX_KDA_HEAD_NUM, "H and HV must be <= 128");
    TORCH_CHECK(kDim >= 16 && kDim <= MAX_KDA_K_DIM && kDim % 16 == 0 && vDim >= 16 && vDim <= 256 && vDim % 16 == 0,
                "K/V must be multiples of 16, K must be <=256, and V must be <=256");

    // ---------------- optional scalar inputs ----------------
    at::Tensor aLogK;
    void *aLogPtr = nullptr;
    const bool hasALog = a_log.has_value() && a_log->defined();
    if (hasALog) {
        TORCH_CHECK(a_log->scalar_type() == at::kFloat, "a_log must be float32");
        aLogK = a_log->contiguous();
        TORCH_CHECK(aLogK.dim() == 1 && aLogK.size(0) == vHeads, "a_log must have shape (HV)");
        aLogPtr = aLogK.data_ptr();
    } else {
        TORCH_CHECK(!use_gate_in_kernel, "a_log is required when use_gate_in_kernel is true");
    }

    at::Tensor dtBiasK;
    void *dtBiasPtr = nullptr;
    const bool hasDtBias = dt_bias.has_value() && dt_bias->defined();
    if (hasDtBias) {
        TORCH_CHECK(dt_bias->scalar_type() == at::kFloat, "dt_bias must be float32");
        dtBiasK = dt_bias->contiguous();
        TORCH_CHECK(dtBiasK.dim() == 1 && dtBiasK.size(0) == vHeads * kDim, "dt_bias must have shape (HV*K)");
        dtBiasPtr = dtBiasK.data_ptr();
    }
    if (safe_gate && use_gate_in_kernel) {
        TORCH_CHECK(lowerBoundValue >= -5.0f && lowerBoundValue < 0.0f,
                    "lower_bound must be in [-5, 0) when safe_gate is true");
    }

    // ---------------- varlen metadata ----------------
    at::Tensor cuSeqlensDev;
    void *cuSeqlensPtr = nullptr;
    std::vector<int64_t> cuHost;
    const bool hasCuSeqlens = cu_seqlens.has_value() && cu_seqlens->defined();
    if (hasCuSeqlens) {
        TORCH_CHECK(cu_seqlens->scalar_type() == at::kLong, "cu_seqlens must be int64");
        TORCH_CHECK(cu_seqlens->dim() == 1 && cu_seqlens->size(0) >= 2,
                    "cu_seqlens must be 1-D with at least two elements");
        cuHost = ReadHostInt64(cu_seqlens->contiguous());
        TORCH_CHECK(cuHost.front() == 0, "cu_seqlens[0] must be 0");
        TORCH_CHECK(cuHost.back() <= seqlen, "cu_seqlens last element must not exceed the sequence length");
        TORCH_CHECK(static_cast<int64_t>(cuHost.size()) - 1 <= MAX_KDA_VARLEN_SEQUENCES,
                    "varlen input supports at most 1024 sequences");
        cuSeqlensDev = cu_seqlens->contiguous();
        if (cuSeqlensDev.device().type() != c10::DeviceType::PrivateUse1) {
            cuSeqlensDev = cuSeqlensDev.to(q.device());
        }
        cuSeqlensPtr = cuSeqlensDev.data_ptr();
    }
    bool isVarLen = false;
    int64_t seqNum = 0;
    int64_t totalChunks = 0;
    TORCH_CHECK(ResolveSequenceInfo(cuHost, seqlen, chunk_size, batch, isVarLen, seqNum, totalChunks),
                "invalid cu_seqlens or degenerate sequence info");
    if (hasCuSeqlens && !isRank3) {
        TORCH_CHECK(batch == 1, "rank-4 varlen input with cu_seqlens requires B=1");
    }

    at::Tensor chunkIndicesDev;
    void *chunkIndicesPtr = nullptr;
    const bool hasChunkIndices = chunk_indices.has_value() && chunk_indices->defined();
    if (hasChunkIndices) {
        TORCH_CHECK(hasCuSeqlens, "chunk_indices requires cu_seqlens");
        TORCH_CHECK(chunk_indices->scalar_type() == at::kLong, "chunk_indices must be int64");
        TORCH_CHECK(chunk_indices->dim() == 1 && chunk_indices->numel() == totalChunks * 2,
                    "chunk_indices must contain exactly one (seq_id, chunk_id) pair per chunk");
        chunkIndicesDev = chunk_indices->contiguous();
        if (chunkIndicesDev.device().type() != c10::DeviceType::PrivateUse1) {
            chunkIndicesDev = chunkIndicesDev.to(q.device());
        }
        chunkIndicesPtr = chunkIndicesDev.data_ptr();
    }

    // ---------------- initial state ----------------
    at::Tensor initialStateK;
    void *initialStatePtr = nullptr;
    const bool hasInitialState = initial_state.has_value() && initial_state->defined();
    if (hasInitialState) {
        TORCH_CHECK(initial_state->scalar_type() == at::kFloat, "initial_state must be float32");
        TORCH_CHECK(initial_state->dim() == 4 && initial_state->size(0) == seqNum &&
                        initial_state->size(1) == vHeads,
                    "initial_state must be (N, HV, K, V) when state_v_first=false and (N, HV, V, K) otherwise");
        if (state_v_first) {
            TORCH_CHECK(initial_state->size(2) == vDim && initial_state->size(3) == kDim,
                        "initial_state must be (N, HV, V, K) when state_v_first=true");
            initialStateK = initial_state->transpose(2, 3).contiguous();
        } else {
            TORCH_CHECK(initial_state->size(2) == kDim && initial_state->size(3) == vDim,
                        "initial_state must be (N, HV, K, V) when state_v_first=false");
            initialStateK = initial_state->contiguous();
        }
        initialStatePtr = initialStateK.data_ptr();
    }

    // ---------------- outputs ----------------
    at::Tensor attnOut = at::empty({batch, seqlen, vHeads, vDim}, q.options());
    at::Tensor aqk = at::empty({batch, vHeads, seqlen, chunk_size}, q.options());
    at::Tensor akk = at::empty({batch, vHeads, seqlen, chunk_size}, q.options());

    at::Tensor finalStateInternal;
    at::Tensor finalStateOut;
    if (output_final_state) {
        finalStateInternal = at::empty({seqNum, vHeads, kDim, vDim}, g.options().dtype(at::kFloat));
        if (state_v_first) {
            finalStateOut = at::empty({seqNum, vHeads, vDim, kDim}, g.options().dtype(at::kFloat));
        } else {
            finalStateOut = finalStateInternal;
        }
    }

    at::Tensor gkOut;
    if (output_gk) {
        gkOut = at::empty({batch, vHeads, seqlen, kDim}, g.options().dtype(at::kFloat));
    }

    at::Tensor wOut;
    if (output_w) {
        wOut = at::empty({batch, vHeads, seqlen, kDim}, q.options());
    }
    at::Tensor uOut;
    if (output_u) {
        uOut = at::empty({batch, vHeads, seqlen, vDim}, q.options());
    }
    at::Tensor qgOut;
    if (output_qg) {
        qgOut = at::empty({batch, vHeads, seqlen, kDim}, q.options());
    }
    at::Tensor kgOut;
    if (output_kg) {
        kgOut = at::empty({batch, vHeads, seqlen, kDim}, q.options());
    }
    at::Tensor vNewOut;
    if (output_v_new) {
        vNewOut = at::empty({batch, vHeads, seqlen, vDim}, q.options());
    }
    at::Tensor hInternal;
    at::Tensor hOut;
    if (output_h) {
        hInternal = at::empty({batch, vHeads, totalChunks, kDim, vDim}, q.options());
        if (state_v_first) {
            hOut = at::empty({batch, totalChunks, vHeads, vDim, kDim}, q.options());
        } else {
            hOut = at::empty({batch, totalChunks, vHeads, kDim, vDim}, q.options());
        }
    }

    // ---------------- platform / tiling ----------------
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    const bool isAscend950 = ascendcPlatform->GetCurNpuArch() == NpuArch::DAV_3510;
    const uint32_t physicalCoreNum = std::max<uint32_t>(ascendcPlatform->GetCoreNumAic(), 1);
    const uint64_t fwdHTaskCount = static_cast<uint64_t>(isVarLen ? seqNum : batch) * vHeads;
    const uint32_t blockDim =
        isAscend950 ?
            static_cast<uint32_t>(std::min<uint64_t>(physicalCoreNum, std::max<uint64_t>(fwdHTaskCount, 1))) :
            physicalCoreNum;
    TORCH_CHECK(blockDim > 0, "invalid AIC core number");

    const auto arch35Options = optiling::arch35::ConfigureChunkKdaFwdArch35(
        isAscend950, q.scalar_type() == at::kBFloat16, g.scalar_type() == at::kFloat, hasALog,
        use_gate_in_kernel, safe_gate, isVarLen, seqlen, vHeads, chunk_size, kDim, vDim, output_qg,
        output_v_new, output_h);

    const int64_t dataBytes = 2;  // bfloat16
    KdaForward::ChunkKdaFwdTilingData tilingData;
    const uint64_t totalWorkspace =
        ComputeTilingData(batch, seqNum, qHeads, vHeads, seqlen, kDim, vDim, chunk_size, totalChunks,
                          sequenceMajor, scaleValue, lowerBoundValue, hasInitialState, isVarLen, safe_gate,
                          use_gate_in_kernel, hasALog, hasDtBias, arch35Options, gateDataType, qDataType,
                          betaDataType, dataBytes, blockDim, tilingData);

    // ---------------- launch ----------------
    constexpr uint32_t PADDING_BYTE = 32;
    const int32_t tilingSize =
        (static_cast<int32_t>(sizeof(KdaForward::ChunkKdaFwdTilingData)) + PADDING_BYTE - 1) / PADDING_BYTE *
        PADDING_BYTE;
    at::Tensor cpuTiling = at::empty({tilingSize}, at::kByte);
    std::memcpy(cpuTiling.data_ptr(), &tilingData, sizeof(KdaForward::ChunkKdaFwdTilingData));
    at::Tensor tilingTensor = TorchNpuHelper::CopyTensorHostToDevice(cpuTiling);

    at::Tensor workspaceTensor = at::empty({static_cast<int64_t>(SYS_WORKSPACE_SIZE + totalWorkspace)},
                                           at::TensorOptions().dtype(at::kByte).device(q.device()));

    void *finalStatePtr = finalStateInternal.defined() ? finalStateInternal.data_ptr() : nullptr;
    void *gkPtr = gkOut.defined() ? gkOut.data_ptr() : nullptr;
    void *wPtr = wOut.defined() ? wOut.data_ptr() : nullptr;
    void *uPtr = uOut.defined() ? uOut.data_ptr() : nullptr;
    void *qgPtr = qgOut.defined() ? qgOut.data_ptr() : nullptr;
    void *kgPtr = kgOut.defined() ? kgOut.data_ptr() : nullptr;
    void *vNewPtr = vNewOut.defined() ? vNewOut.data_ptr() : nullptr;
    void *hPtr = hInternal.defined() ? hInternal.data_ptr() : nullptr;

    EXEC_KERNEL_CMD(chunk_kda_fwd, blockDim, qK, kK, vK, gK, betaK, aLogPtr, dtBiasPtr, initialStatePtr,
                    cuSeqlensPtr, chunkIndicesPtr, attnOut, finalStatePtr, gkPtr, aqk, akk, wPtr, uPtr, qgPtr,
                    kgPtr, vNewPtr, hPtr, workspaceTensor, tilingTensor);

    // ---------------- export transposed views ----------------
    if (output_final_state && state_v_first) {
        finalStateOut.copy_(finalStateInternal.transpose(2, 3));
    }
    if (output_h) {
        hOut.copy_(state_v_first ? hInternal.permute({0, 2, 1, 4, 3}) : hInternal.permute({0, 2, 1, 3, 4}));
    }

    return std::make_tuple(attnOut, finalStateOut, gkOut, aqk, akk, wOut, uOut, qgOut, kgOut, vNewOut, hOut);
}

}  // namespace npu_kernel
}  // namespace sglang
