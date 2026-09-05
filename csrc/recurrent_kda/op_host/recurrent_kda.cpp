/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the sgl-kernel-npu project
 */

/*!
 * \file recurrent_kda.cpp
 * \brief Host-side direct-launch (non-aclnn) implementation of the fused
 * recurrent KDA AscendC kernel, migrated from vllm-ascend.
 */

#include <cstring>
#include <limits>

#include "acl/acl.h"
#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/DeviceUtils.h"
#include "tiling/platform/platform_ascendc.h"

#include "defines.h"
#include "torch_helper.h"
#include "../op_kernel/recurrent_kda_struct.h"
#include "stub/aclrtlaunch_recurrent_kda.h"

namespace sglang {
namespace npu_kernel {

namespace {

using RecurrentKdaTilingData = RecurrentKda::RecurrentKdaTilingData;

constexpr size_t RKDA_STATE_DIM_NUM = 4;
constexpr size_t RKDA_DIM_0 = 0;
constexpr size_t RKDA_DIM_1 = 1;
constexpr size_t RKDA_DIM_2 = 2;
constexpr size_t RKDA_DIM_3 = 3;
constexpr uint32_t RKDA_LAYOUT_BSND = 0;
constexpr uint32_t RKDA_LAYOUT_TND = 1;
constexpr size_t RKDA_MAX_MTP = 8;
constexpr int64_t RKDA_UB_GUARD_BYTES = 2048;
constexpr size_t RKDA_SYS_WORKSPACE_SIZE = 16U * 1024U * 1024U;
constexpr int64_t PADDING_BYTE = 32;
constexpr int64_t MAX_CAPTURE_NUM = 1024;

constexpr uint32_t DT_FLOAT_GE = 0;
constexpr uint32_t DT_FLOAT16_GE = 1;
constexpr uint32_t DT_BF16_GE = 2;

inline int64_t CeilDiv(int64_t x, int64_t y)
{
    return (x + y - 1) / y;
}

inline int64_t CeilAlign(int64_t value, int64_t align)
{
    return CeilDiv(value, align) * align;
}

int64_t StateDtypeSize(const at::ScalarType &dtype)
{
    return (dtype == at::kFloat) ? 4 : 2;
}

uint32_t GeDtypeCode(const at::ScalarType &dtype)
{
    if (dtype == at::kFloat) {
        return DT_FLOAT_GE;
    }
    if (dtype == at::kHalf) {
        return DT_FLOAT16_GE;
    }
    if (dtype == at::kBFloat16) {
        return DT_BF16_GE;
    }
    return DT_FLOAT_GE;
}

uint32_t IndexDtypeCode(const at::ScalarType &dtype)
{
    return (dtype == at::kInt) ? 0 : 1;
}

struct BufferProfile {
    uint32_t stateOutBufferNum = 1;
    uint32_t attnOutBufferNum = 1;
    uint32_t vStep = 0;
    uint32_t repeatTime = 0;
    bool valid = false;
};

int64_t CalcFixedUbBytes(int64_t aNv, int64_t aDv, int64_t aDk)
{
    int64_t usedUbBytes = 0;
    usedUbBytes += static_cast<int64_t>(RKDA_MAX_MTP) * (4 * aDk + 2 * aDv);
    usedUbBytes += static_cast<int64_t>(RKDA_MAX_MTP) * 4 * aDk;
    usedUbBytes += static_cast<int64_t>(RKDA_MAX_MTP) * 4 * aNv;
    usedUbBytes += 64 + RKDA_UB_GUARD_BYTES;
    return usedUbBytes;
}

int64_t CalcWorkingUbBytes(int64_t aNv, int64_t aDv, int64_t aDk)
{
    int64_t usedUbBytes = CalcFixedUbBytes(aNv, aDv, aDk);
    usedUbBytes += static_cast<int64_t>(RKDA_MAX_MTP) * (4 * aDv + 12 * aDk + 4 * aNv);
    return usedUbBytes;
}

int64_t CalcVStepCoeff(int64_t aDk, int64_t stateDtypeSize, uint32_t stateOutBufferNum,
                       uint32_t attnOutBufferNum)
{
    int64_t coeff = stateDtypeSize * aDk;
    coeff += static_cast<int64_t>(stateOutBufferNum) * stateDtypeSize * aDk;
    coeff += static_cast<int64_t>(attnOutBufferNum) * 2;
    coeff += 8 * aDk + 8;
    return coeff;
}

bool EvaluateBufferProfile(int64_t ubSize, int64_t usedUbBytes, int64_t aDk, int64_t stateDtypeSize,
                           uint32_t stateOutBufferNum, uint32_t attnOutBufferNum, uint32_t dv,
                           BufferProfile &profile)
{
    int64_t coeff = CalcVStepCoeff(aDk, stateDtypeSize, stateOutBufferNum, attnOutBufferNum);
    int64_t vStep = (ubSize - usedUbBytes) / coeff / 8 * 8;
    if (vStep < static_cast<int64_t>(RKDA_MAX_MTP)) {
        return false;
    }
    int64_t repeatTime = CeilDiv(dv, static_cast<uint32_t>(vStep));
    vStep = CeilAlign(CeilDiv(dv, static_cast<uint32_t>(repeatTime)), static_cast<int64_t>(8));
    if (vStep < static_cast<int64_t>(RKDA_MAX_MTP)) {
        return false;
    }
    profile.stateOutBufferNum = stateOutBufferNum;
    profile.attnOutBufferNum = attnOutBufferNum;
    profile.vStep = static_cast<uint32_t>(vStep);
    profile.repeatTime = static_cast<uint32_t>(repeatTime);
    profile.valid = true;
    return true;
}

bool IsBetterProfile(const BufferProfile &candidate, const BufferProfile &current)
{
    if (!current.valid) {
        return true;
    }
    if (candidate.repeatTime != current.repeatTime) {
        return candidate.repeatTime < current.repeatTime;
    }
    uint32_t candidateDepth = candidate.stateOutBufferNum + candidate.attnOutBufferNum;
    uint32_t currentDepth = current.stateOutBufferNum + current.attnOutBufferNum;
    if (candidateDepth != currentDepth) {
        return candidateDepth > currentDepth;
    }
    return candidate.vStep > current.vStep;
}

}  // namespace

HOST_API at::Tensor recurrent_kda_impl(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
    const at::Tensor &gate, const at::Tensor &beta, at::Tensor &initial_state,
    const at::Tensor &cu_seqlens, const at::Tensor &ssm_state_indices,
    const at::Tensor &a_log, const at::Tensor &dt_bias,
    const c10::optional<at::Tensor> &num_accepted_tokens, double scale,
    bool use_qk_l2norm_in_kernel, bool use_gate_in_kernel, bool use_beta_sigmoid_in_kernel,
    bool allow_neg_eigval, bool safe_gate, double lower_bound)
{
    const bool is_tnd = query.dim() == 3;
    TORCH_CHECK((is_tnd && key.dim() == 3 && value.dim() == 3 && gate.dim() == 3 && beta.dim() == 2) ||
                    (!is_tnd && query.dim() == 4 && key.dim() == 4 && value.dim() == 4 &&
                     gate.dim() == 4 && beta.dim() == 3),
                "recurrent_kda: TND expects q/k [T,H,K], v [T,HV,V], gate [T,HV,K], beta [T,HV]; "
                "BSND expects q/k [B,T,H,K], v [B,T,HV,V], gate [B,T,HV,K], beta [B,T,HV].");
    TORCH_CHECK(query.sizes() == key.sizes(),
                "recurrent_kda: query and key must have identical shapes.");
    TORCH_CHECK(query.scalar_type() == at::kBFloat16 &&
                    key.scalar_type() == at::kBFloat16 &&
                    value.scalar_type() == at::kBFloat16,
                "recurrent_kda: query/key/value must be bfloat16.");
    TORCH_CHECK((gate.scalar_type() == at::kFloat || gate.scalar_type() == at::kBFloat16 ||
                 gate.scalar_type() == at::kHalf) &&
                    (beta.scalar_type() == at::kFloat || beta.scalar_type() == at::kBFloat16 ||
                     beta.scalar_type() == at::kHalf),
                "recurrent_kda: gate and beta must be float32, bfloat16 or float16.");
    TORCH_CHECK(key.device() == query.device() && value.device() == query.device() &&
                    gate.device() == query.device() && beta.device() == query.device() &&
                    initial_state.device() == query.device(),
                "recurrent_kda: query/key/value/gate/beta/state must be on the same device.");
    TORCH_CHECK(cu_seqlens.dim() == 1 && cu_seqlens.numel() >= 2,
                "recurrent_kda: cu_seqlens must be a 1D device tensor with at least two elements.");
    TORCH_CHECK(cu_seqlens.scalar_type() == at::kInt || cu_seqlens.scalar_type() == at::kLong,
                "recurrent_kda: cu_seqlens must be int32 or int64.");
    TORCH_CHECK(cu_seqlens.device() == query.device(),
                "recurrent_kda: cu_seqlens must be on the same device as query.");

    const int64_t batch = is_tnd ? 1 : query.size(0);
    const int64_t total_tokens = is_tnd ? query.size(0) : query.size(0) * query.size(1);
    const int64_t seq_num = cu_seqlens.size(0) - 1;
    const int64_t h = is_tnd ? query.size(1) : query.size(2);
    const int64_t k_dim = is_tnd ? query.size(2) : query.size(3);
    const int64_t hv = is_tnd ? value.size(1) : value.size(2);
    const int64_t v_dim = is_tnd ? value.size(2) : value.size(3);
    TORCH_CHECK(total_tokens > 0 && h > 0 && hv > 0,
                "recurrent_kda: token and head dimensions must be positive.");
    TORCH_CHECK(hv % h == 0, "recurrent_kda: HV must be divisible by H.");
    TORCH_CHECK(k_dim == 128 && (v_dim == 128 || v_dim == 256),
                "recurrent_kda: the Kimi K3 integration requires K=128 and V=128 or 256.");
    TORCH_CHECK((is_tnd && value.size(0) == total_tokens && gate.size(0) == total_tokens &&
                 beta.size(0) == total_tokens && gate.size(1) == hv && gate.size(2) == k_dim &&
                 beta.size(1) == hv) ||
                    (!is_tnd && value.size(0) == batch && value.size(1) == query.size(1) &&
                     gate.size(0) == batch && gate.size(1) == query.size(1) && gate.size(2) == hv &&
                     gate.size(3) == k_dim && beta.size(0) == batch && beta.size(1) == query.size(1) &&
                     beta.size(2) == hv),
                "recurrent_kda: value/gate/beta shapes do not match the selected layout.");

    const bool packed_indices = ssm_state_indices.dim() == 1 &&
                                ssm_state_indices.numel() >= total_tokens;
    const bool speculative_indices = ssm_state_indices.dim() == 2 &&
                                     ssm_state_indices.size(0) == seq_num &&
                                     ssm_state_indices.size(1) > 0;
    TORCH_CHECK((ssm_state_indices.scalar_type() == at::kInt ||
                 ssm_state_indices.scalar_type() == at::kLong) &&
                    (packed_indices || speculative_indices),
                "recurrent_kda: ssm_state_indices must be int32/int64 packed [T] or "
                "speculative [seq_num,max_step].");
    TORCH_CHECK(ssm_state_indices.device() == query.device(),
                "recurrent_kda: ssm_state_indices must be on the same device as query.");
    TORCH_CHECK(initial_state.dim() == 4 && initial_state.size(0) >= 1 &&
                    initial_state.size(1) == hv && initial_state.size(2) == v_dim &&
                    initial_state.size(3) == k_dim,
                "recurrent_kda: initial_state must be a non-empty [state_capacity,HV,V,K] pool.");
    TORCH_CHECK(initial_state.scalar_type() == at::kFloat || initial_state.scalar_type() == at::kBFloat16,
                "recurrent_kda: initial_state must be float32 or bfloat16.");
    TORCH_CHECK(a_log.scalar_type() == at::kFloat && a_log.dim() == 1 && a_log.numel() == hv,
                "recurrent_kda: A_log must be float32 [HV].");
    TORCH_CHECK(dt_bias.scalar_type() == at::kFloat &&
                    ((dt_bias.dim() == 1 && dt_bias.numel() == hv * k_dim) ||
                     (dt_bias.dim() == 2 && dt_bias.size(0) == hv && dt_bias.size(1) == k_dim)),
                "recurrent_kda: dt_bias must be float32 [HV*K] or [HV,K].");
    TORCH_CHECK(a_log.device() == query.device() && dt_bias.device() == query.device(),
                "recurrent_kda: A_log and dt_bias must be on the same device as query.");
    if (num_accepted_tokens.has_value() && num_accepted_tokens->defined()) {
        TORCH_CHECK(num_accepted_tokens->dim() == 1 && num_accepted_tokens->size(0) == seq_num &&
                        (num_accepted_tokens->scalar_type() == at::kInt ||
                         num_accepted_tokens->scalar_type() == at::kLong),
                    "recurrent_kda: num_accepted_tokens must be int32/int64 [seq_num].");
        TORCH_CHECK(num_accepted_tokens->device() == query.device(),
                    "recurrent_kda: num_accepted_tokens must be on the same device as query.");
    }
    TORCH_CHECK(!safe_gate || (lower_bound >= -5.0 && lower_bound < 0.0),
                "recurrent_kda: lower_bound must be in [-5,0) for safe gate.");

    const at::Tensor &query_c = query.contiguous();
    const at::Tensor &key_c = key.contiguous();
    const at::Tensor &value_c = value.contiguous();
    const at::Tensor &gate_c = gate.contiguous();
    const at::Tensor &beta_c = beta.contiguous();
    TORCH_CHECK(initial_state.is_contiguous(), "recurrent_kda: initial_state must be contiguous");
    at::Tensor &initial_state_ref = const_cast<at::Tensor &>(initial_state);
    const at::Tensor cu_seqlens_c = cu_seqlens.contiguous();
    const at::Tensor ssm_state_indices_c = ssm_state_indices.contiguous();
    const at::Tensor a_log_c = a_log.contiguous();
    const at::Tensor dt_bias_c = dt_bias.contiguous();

    at::Tensor num_accepted_tokens_c;
    bool has_accepted_tokens = false;
    if (num_accepted_tokens.has_value() && num_accepted_tokens->defined()) {
        has_accepted_tokens = true;
        num_accepted_tokens_c = num_accepted_tokens->contiguous();
    } else {
        num_accepted_tokens_c = at::empty({0}, query.options().dtype(at::kInt));
    }

    at::Tensor output = at::empty_like(value_c);

    // ---------------- tiling ----------------
    auto ascendc_platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint64_t ubSize = 0;
    ascendc_platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    uint32_t aivNum = static_cast<uint32_t>(ascendc_platform->GetCoreNumAiv());
    if (aivNum <= 0) {
        aivNum = 1;
    }

    RecurrentKdaTilingData td;
    std::memset(&td, 0, sizeof(td));

    td.layout = is_tnd ? RKDA_LAYOUT_TND : RKDA_LAYOUT_BSND;
    td.scale = static_cast<float>(scale);
    td.lowerBound = static_cast<float>(lower_bound);
    td.useQkL2norm = use_qk_l2norm_in_kernel ? 1 : 0;
    td.useGateInKernel = use_gate_in_kernel ? 1 : 0;
    td.useBetaSigmoid = use_beta_sigmoid_in_kernel ? 1 : 0;
    td.allowNegEigval = allow_neg_eigval ? 1 : 0;
    td.safeGate = safe_gate ? 1 : 0;
    td.stateVFirst = 1;
    td.outputFinalState = 0;
    td.inplaceFinalState = 1;
    td.hasCuSeqlens = 1;
    td.hasSsmStateIndices = 1;
    td.hasALog = use_gate_in_kernel ? 1 : 0;
    td.hasDtBias = use_gate_in_kernel ? 1 : 0;
    td.hasAcceptedTokens = has_accepted_tokens ? 1 : 0;
    td.gateDtype = GeDtypeCode(gate_c.scalar_type());
    td.betaDtype = GeDtypeCode(beta_c.scalar_type());
    td.cuSeqlensDtype = IndexDtypeCode(cu_seqlens_c.scalar_type());
    td.ssmStateIndicesDtype = IndexDtypeCode(ssm_state_indices_c.scalar_type());
    td.acceptedTokensDtype = has_accepted_tokens ? IndexDtypeCode(num_accepted_tokens_c.scalar_type()) : 1;

    if (is_tnd) {
        td.t = static_cast<uint32_t>(query_c.size(0));
        td.seqLen = static_cast<uint32_t>(query_c.size(0));
        td.nk = static_cast<uint32_t>(query_c.size(1));
        td.dk = static_cast<uint32_t>(query_c.size(2));
        td.nv = static_cast<uint32_t>(value_c.size(1));
        td.dv = static_cast<uint32_t>(value_c.size(2));
        td.b = static_cast<uint32_t>(seq_num);
    } else {
        td.seqLen = static_cast<uint32_t>(query_c.size(1));
        td.t = static_cast<uint32_t>(query_c.size(0) * query_c.size(1));
        td.nk = static_cast<uint32_t>(query_c.size(2));
        td.dk = static_cast<uint32_t>(query_c.size(3));
        td.nv = static_cast<uint32_t>(value_c.size(2));
        td.dv = static_cast<uint32_t>(value_c.size(3));
        td.b = static_cast<uint32_t>(seq_num);
    }
    td.sBlockNum = static_cast<uint32_t>(initial_state_ref.size(0));
    td.ssmStateStride =
        (speculative_indices) ? static_cast<uint32_t>(ssm_state_indices_c.size(1)) : 0;

    const int64_t stateDtypeSize = StateDtypeSize(initial_state_ref.scalar_type());
    const auto &strides = initial_state_ref.strides();
    td.stateInStride0 = static_cast<uint64_t>(strides[RKDA_DIM_0]);
    td.stateInStride1 = static_cast<uint64_t>(strides[RKDA_DIM_1]);
    td.stateInStride2 = static_cast<uint64_t>(strides[RKDA_DIM_2]);
    td.stateInStride3 = static_cast<uint64_t>(strides[RKDA_DIM_3]);
    td.stateOutStride0 = td.stateInStride0;
    td.stateOutStride1 = td.stateInStride1;
    td.stateOutStride2 = td.stateInStride2;
    td.stateOutStride3 = td.stateInStride3;

    // block dim: min(b*nv, aivNum)
    uint64_t taskUnits = static_cast<uint64_t>(td.b) * static_cast<uint64_t>(td.nv);
    if (taskUnits == 0) {
        taskUnits = 1;
    }
    uint64_t selectedCoreNum = (taskUnits < aivNum) ? taskUnits : aivNum;
    td.vectorCoreNum = static_cast<uint32_t>(selectedCoreNum);

    // UB budget -> vStep and buffer counts.
    int64_t aNv = CeilAlign(td.nv, 16);
    int64_t aDv = CeilAlign(td.dv, 16);
    int64_t aDk = CeilAlign(td.dk, 16);
    int64_t fixedUbBytes = CalcFixedUbBytes(aNv, aDv, aDk);
    int64_t workingUbBytes = CalcWorkingUbBytes(aNv, aDv, aDk);

    uint32_t vStep = 0;
    uint32_t stateOutBufferNum = 1;
    uint32_t attnOutBufferNum = 1;
    int64_t ubRestBytes = 0;

    BufferProfile selected;
    const BufferProfile candidates[3] = {
        {1, 1, 0, 0, false},
        {1, 2, 0, 0, false},
        {2, 2, 0, 0, false},
    };
    for (const auto &candidate : candidates) {
        BufferProfile profile;
        if (!EvaluateBufferProfile(static_cast<int64_t>(ubSize), workingUbBytes, aDk, stateDtypeSize,
                                   candidate.stateOutBufferNum, candidate.attnOutBufferNum, td.dv, profile)) {
            continue;
        }
        if (IsBetterProfile(profile, selected)) {
            selected = profile;
        }
    }
    TORCH_CHECK(selected.valid, "recurrent_kda: vStep should be at least 8, shape is too big");

    int64_t queueCoeff = CalcVStepCoeff(aDk, stateDtypeSize, selected.stateOutBufferNum,
                                        selected.attnOutBufferNum) -
                         (8 * aDk + 8);
    ubRestBytes = static_cast<int64_t>(ubSize) - fixedUbBytes -
                  queueCoeff * static_cast<int64_t>(selected.vStep);
    TORCH_CHECK(ubRestBytes >= 0, "recurrent_kda: ubRestBytes should be non-negative");

    td.ubCalSize = static_cast<uint32_t>(ubSize);
    td.ubRestBytes = static_cast<uint32_t>(ubRestBytes);
    td.vStep = selected.vStep;
    td.stateOutBufferNum = selected.stateOutBufferNum;
    td.attnOutBufferNum = selected.attnOutBufferNum;

    // ---------------- tiling buffer & workspace ----------------
    int32_t tilingSize = static_cast<int32_t>((sizeof(RecurrentKdaTilingData) + PADDING_BYTE - 1) /
                                              PADDING_BYTE * PADDING_BYTE);

    auto copyTilingToDevice = [&]() {
        auto cpuTiling = at::empty({tilingSize}, at::kByte);
        std::memcpy(cpuTiling.data_ptr(), &td, sizeof(RecurrentKdaTilingData));
        return TorchNpuHelper::CopyTensorHostToDevice(cpuTiling);
    };

    static auto globalTilingBuffer = at::empty({tilingSize * MAX_CAPTURE_NUM},
                                               at::TensorOptions().dtype(at::kByte).device(query.device()));
    static int64_t gCaptureNum = 0;

    at::Tensor tilingTensor;
    if (gCaptureNum >= MAX_CAPTURE_NUM) {
        tilingTensor = copyTilingToDevice();
    } else {
        auto deviceTiling = copyTilingToDevice();
        globalTilingBuffer
            .slice(0, gCaptureNum * tilingSize, gCaptureNum * tilingSize + tilingSize)
            .copy_(deviceTiling);
        tilingTensor = at::from_blob(
            globalTilingBuffer.data_ptr<uint8_t>() + (tilingSize * gCaptureNum),
            tilingSize, at::kByte);
        gCaptureNum++;
    }

    int32_t libApiWorkspaceSize = static_cast<int32_t>(ascendc_platform->GetLibApiWorkSpaceSize());
    int64_t totalWorkspace = std::max(static_cast<int64_t>(libApiWorkspaceSize),
                                      static_cast<int64_t>(RKDA_SYS_WORKSPACE_SIZE));
    auto workspaceTensor = at::empty({totalWorkspace},
                                     at::TensorOptions().dtype(at::kByte).device(query.device()));

    uint32_t blockDim = td.vectorCoreNum;
    if (blockDim <= 0) {
        blockDim = 1;
    }

    EXEC_KERNEL_CMD(recurrent_kda, blockDim, query_c, key_c, value_c, gate_c, beta_c,
                    initial_state_ref, cu_seqlens_c, ssm_state_indices_c, a_log_c, dt_bias_c,
                    num_accepted_tokens_c, output, initial_state_ref, initial_state_ref,
                    workspaceTensor, tilingTensor);

    return output;
}

}  // namespace npu_kernel
}  // namespace sglang