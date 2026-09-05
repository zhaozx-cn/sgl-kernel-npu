/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
 */

#include <cstring>
#include <array>
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
#include "common.h"
#include "aclrtlaunch_recurrent_kda.h"
#include "recurrent_kda.h"
#include "../op_kernel/recurrent_kda_struct.h"

namespace sglang {
namespace npu_kernel {

using RecurrentKdaTilingData = RecurrentKda::RecurrentKdaTilingData;

namespace {

constexpr uint64_t RKDA_MAX_MTP = 8;
constexpr int64_t RKDA_UB_GUARD_BYTES = 2048;
constexpr int64_t RKDA_SYS_WORKSPACE_SIZE = 16LL * 1024LL * 1024LL;
constexpr uint32_t PADDING_BYTE = 32U;
constexpr uint64_t BF16_NUM_PER_BLOCK = 16;
constexpr uint64_t FP32_NUM_PER_BLOCK = 8;
constexpr uint32_t MAX_OUT_BUFFER_NUM = 2;
constexpr uint64_t BUFFER_NUM = 1;

inline int64_t CeilDiv(int64_t x, int64_t y)
{
    return (x + y - 1) / y;
}

inline int64_t CeilAlign(int64_t value, int64_t align)
{
    return CeilDiv(value, align) * align;
}

struct UbCalcContext {
    int64_t ubSize = 0;
    int64_t aNv = 0;
    int64_t aDv = 0;
    int64_t aDk = 0;
    int64_t fixedUbBytes = 0;
    int64_t workingUbBytes = 0;
    int64_t coeff = 0;
};

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

int64_t CalcVStepCoeff(int64_t aDk, bool isStateFp32, uint32_t stateOutBufferNum, uint32_t attnOutBufferNum)
{
    int64_t stateDtypeSize = isStateFp32 ? 4 : 2;
    int64_t coeff = stateDtypeSize * aDk;
    coeff += static_cast<int64_t>(stateOutBufferNum) * stateDtypeSize * aDk;
    coeff += static_cast<int64_t>(attnOutBufferNum) * 2;
    coeff += 8 * aDk + 8;
    return coeff;
}

bool EvaluateBufferProfile(int64_t ubSize, int64_t usedUbBytes, int64_t aDk, bool isStateFp32,
                           uint32_t stateOutBufferNum, uint32_t attnOutBufferNum,
                           uint32_t dv, BufferProfile &profile)
{
    int64_t coeff = CalcVStepCoeff(aDk, isStateFp32, stateOutBufferNum, attnOutBufferNum);
    int64_t vStep = (ubSize - usedUbBytes) / coeff / 8 * 8;
    if (vStep < static_cast<int64_t>(RKDA_MAX_MTP)) {
        return false;
    }
    int64_t repeatTime = CeilDiv(static_cast<int64_t>(dv), vStep);
    vStep = CeilAlign(CeilDiv(static_cast<int64_t>(dv), repeatTime), 8);
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

void ComputeTiling(
    const at::Tensor &query, const at::Tensor &value, const at::Tensor &initial_state,
    const at::Tensor &cu_seqlens, const at::Tensor &ssm_state_indices,
    bool has_cu_seqlens, bool has_ssm_state_indices,
    bool has_a_log, bool has_dt_bias, bool has_accepted_tokens,
    bool use_qk_l2norm, bool use_gate_in_kernel, bool use_beta_sigmoid,
    bool allow_neg_eigval, bool safe_gate, bool state_v_first,
    bool is_tnd, bool is_state_fp32,
    float scale, float lower_bound,
    uint64_t aiv_num, uint64_t ub_size,
    RecurrentKdaTilingData &tiling)
{
    std::memset(&tiling, 0, sizeof(tiling));

    int64_t batch = 0;
    int64_t total_tokens = 0;
    int64_t seq_len = 0;
    int64_t nk = 0;
    int64_t dk = 0;
    int64_t nv = 0;
    int64_t dv = 0;

    if (is_tnd) {
        total_tokens = query.size(0);
        seq_len = total_tokens;
        nk = query.size(1);
        dk = query.size(2);
        nv = value.size(1);
        dv = value.size(2);
        batch = has_cu_seqlens ? (cu_seqlens.size(0) - 1) : 1;
    } else {
        batch = query.size(0);
        seq_len = query.size(1);
        total_tokens = batch * seq_len;
        nk = query.size(2);
        dk = query.size(3);
        nv = value.size(2);
        dv = value.size(3);
    }

    int64_t seq_num = has_cu_seqlens ? (cu_seqlens.size(0) - 1) : (is_tnd ? 1 : batch);
    int64_t state_capacity = initial_state.size(0);

    int64_t ssm_state_stride = 0;
    if (has_ssm_state_indices && ssm_state_indices.dim() == 2) {
        ssm_state_stride = ssm_state_indices.size(1);
    }

    // Compute state strides (contiguous layout)
    int64_t state_dim3 = initial_state.size(3); // K
    int64_t state_dim2 = initial_state.size(2); // V (if state_v_first) or K
    int64_t state_dim1 = initial_state.size(1); // HV
    int64_t state_stride3 = 1;
    int64_t state_stride2 = state_dim3;
    int64_t state_stride1 = state_dim2 * state_stride2;
    int64_t state_stride0 = state_dim1 * state_stride1;

    // Determine gate/beta dtype codes: 0=float, 1=bf16, 2=fp16
    auto dtypeCode = [](at::ScalarType t) -> uint32_t {
        if (t == at::kFloat) return 0;
        if (t == at::kBFloat16) return 1;
        return 2;
    };
    auto intDtypeCode = [](at::ScalarType t) -> uint32_t {
        return (t == at::kInt) ? 0 : 1;
    };

    // Fill basic tiling fields
    tiling.vectorCoreNum = static_cast<uint32_t>(aiv_num);
    tiling.ubCalSize = static_cast<uint32_t>(ub_size);
    tiling.t = static_cast<uint32_t>(total_tokens);
    tiling.seqLen = static_cast<uint32_t>(seq_len);
    tiling.nk = static_cast<uint32_t>(nk);
    tiling.dk = static_cast<uint32_t>(dk);
    tiling.nv = static_cast<uint32_t>(nv);
    tiling.dv = static_cast<uint32_t>(dv);
    tiling.sBlockNum = static_cast<uint32_t>(state_capacity);
    tiling.ssmStateStride = static_cast<uint32_t>(ssm_state_stride);
    tiling.b = static_cast<uint32_t>(seq_num);
    tiling.scale = scale;
    tiling.lowerBound = lower_bound;
    tiling.layout = is_tnd ? 1 : 0; // 0=BSND, 1=TND
    tiling.hasCuSeqlens = has_cu_seqlens ? 1 : 0;
    tiling.hasSsmStateIndices = has_ssm_state_indices ? 1 : 0;
    tiling.hasALog = has_a_log ? 1 : 0;
    tiling.hasDtBias = has_dt_bias ? 1 : 0;
    tiling.hasAcceptedTokens = has_accepted_tokens ? 1 : 0;
    tiling.useQkL2norm = use_qk_l2norm ? 1 : 0;
    tiling.useGateInKernel = use_gate_in_kernel ? 1 : 0;
    tiling.useBetaSigmoid = use_beta_sigmoid ? 1 : 0;
    tiling.allowNegEigval = allow_neg_eigval ? 1 : 0;
    tiling.safeGate = safe_gate ? 1 : 0;
    tiling.stateVFirst = state_v_first ? 1 : 0;
    tiling.outputFinalState = 0;
    tiling.inplaceFinalState = 1;
    // gate/beta dtype will be set by caller from the actual tensor dtypes
    tiling.cuSeqlensDtype = has_cu_seqlens ? intDtypeCode(cu_seqlens.scalar_type()) : 0;
    tiling.ssmStateIndicesDtype = has_ssm_state_indices ? intDtypeCode(ssm_state_indices.scalar_type()) : 0;
    tiling.acceptedTokensDtype = 0;
    tiling.stateInStride0 = static_cast<uint64_t>(state_stride0);
    tiling.stateInStride1 = static_cast<uint64_t>(state_stride1);
    tiling.stateInStride2 = static_cast<uint64_t>(state_stride2);
    tiling.stateInStride3 = static_cast<uint64_t>(state_stride3);
    tiling.stateOutStride0 = static_cast<uint64_t>(state_stride0);
    tiling.stateOutStride1 = static_cast<uint64_t>(state_stride1);
    tiling.stateOutStride2 = static_cast<uint64_t>(state_stride2);
    tiling.stateOutStride3 = static_cast<uint64_t>(state_stride3);

    // Compute dynamic block dim
    uint64_t taskUnits = static_cast<uint64_t>(seq_num) * static_cast<uint64_t>(nv);
    if (taskUnits == 0) {
        taskUnits = 1;
    }
    uint64_t maxCoreNum = (aiv_num > 0) ? aiv_num : 1;
    uint64_t selectedCoreNum = (taskUnits < maxCoreNum) ? taskUnits : maxCoreNum;
    tiling.vectorCoreNum = static_cast<uint32_t>(selectedCoreNum);

    // Compute UB and vStep
    int64_t aNv = CeilAlign(nv, static_cast<int64_t>(BF16_NUM_PER_BLOCK));
    int64_t aDv = CeilAlign(dv, static_cast<int64_t>(BF16_NUM_PER_BLOCK));
    int64_t aDk = CeilAlign(dk, static_cast<int64_t>(BF16_NUM_PER_BLOCK));

    int64_t fixedUbBytes = CalcFixedUbBytes(aNv, aDv, aDk);
    int64_t workingUbBytes = CalcWorkingUbBytes(aNv, aDv, aDk);

    BufferProfile selected;
    const std::array<BufferProfile, 3> candidates = {{
        {1, 1, 0, 0, false},
        {1, 2, 0, 0, false},
        {2, 2, 0, 0, false},
    }};
    for (const auto &candidate : candidates) {
        BufferProfile profile;
        if (!EvaluateBufferProfile(static_cast<int64_t>(ub_size), workingUbBytes, aDk, is_state_fp32,
                                    candidate.stateOutBufferNum, candidate.attnOutBufferNum,
                                    static_cast<uint32_t>(dv), profile)) {
            continue;
        }
        if (IsBetterProfile(profile, selected)) {
            selected = profile;
        }
    }

    TORCH_CHECK(selected.valid, "recurrent_kda: vStep should be at least 8, shape is too big");

    int64_t queueCoeff = CalcVStepCoeff(aDk, is_state_fp32, selected.stateOutBufferNum, selected.attnOutBufferNum)
                         - (8 * aDk + 8);
    int64_t ubRestBytes = static_cast<int64_t>(ub_size) - fixedUbBytes
                          - queueCoeff * static_cast<int64_t>(selected.vStep);
    TORCH_CHECK(ubRestBytes >= 0, "recurrent_kda: ubRestBytes should be non-negative");

    tiling.ubRestBytes = static_cast<uint32_t>(ubRestBytes);
    tiling.vStep = selected.vStep;
    tiling.stateOutBufferNum = selected.stateOutBufferNum;
    tiling.attnOutBufferNum = selected.attnOutBufferNum;
}

}  // namespace

HOST_API at::Tensor recurrent_kda_impl(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
    const at::Tensor &gate, const at::Tensor &beta, at::Tensor &initial_state,
    const at::Tensor &cu_seqlens, const at::Tensor &ssm_state_indices,
    const c10::optional<at::Tensor> &a_log_opt, const c10::optional<at::Tensor> &dt_bias_opt,
    const c10::optional<at::Tensor> &num_accepted_tokens_opt,
    double scale, bool use_qk_l2norm_in_kernel, bool use_gate_in_kernel,
    bool use_beta_sigmoid_in_kernel, bool allow_neg_eigval,
    bool safe_gate, double lower_bound, bool state_v_first)
{
    TORCH_CHECK(query.defined(), "recurrent_kda: query must be defined");
    TORCH_CHECK(key.defined(), "recurrent_kda: key must be defined");
    TORCH_CHECK(value.defined(), "recurrent_kda: value must be defined");
    TORCH_CHECK(gate.defined(), "recurrent_kda: gate must be defined");
    TORCH_CHECK(beta.defined(), "recurrent_kda: beta must be defined");
    TORCH_CHECK(initial_state.defined(), "recurrent_kda: initial_state must be defined");

    TORCH_CHECK(query.scalar_type() == at::kBFloat16,
                "recurrent_kda: query must be bfloat16");
    TORCH_CHECK(key.scalar_type() == at::kBFloat16,
                "recurrent_kda: key must be bfloat16");
    TORCH_CHECK(value.scalar_type() == at::kBFloat16,
                "recurrent_kda: value must be bfloat16");

    bool is_tnd = query.dim() == 3;
    bool is_state_fp32 = (initial_state.scalar_type() == at::kFloat);

    bool has_cu_seqlens = cu_seqlens.defined() && cu_seqlens.numel() > 0;
    bool has_ssm_state_indices = ssm_state_indices.defined() && ssm_state_indices.numel() > 0;
    bool has_a_log = a_log_opt.has_value() && a_log_opt.value().defined() && a_log_opt.value().numel() > 0;
    bool has_dt_bias = dt_bias_opt.has_value() && dt_bias_opt.value().defined() && dt_bias_opt.value().numel() > 0;
    bool has_accepted_tokens = num_accepted_tokens_opt.has_value() && num_accepted_tokens_opt.value().defined()
                               && num_accepted_tokens_opt.value().numel() > 0;

    int devidx = query.device().index();
    c10_npu::set_device(devidx);

    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint64_t ubSize{0};
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    uint64_t aivNum = ascendcPlatform->GetCoreNumAiv();

    RecurrentKdaTilingData tiling;
    ComputeTiling(query, value, initial_state, cu_seqlens, ssm_state_indices,
                  has_cu_seqlens, has_ssm_state_indices,
                  has_a_log, has_dt_bias, has_accepted_tokens,
                  use_qk_l2norm_in_kernel, use_gate_in_kernel, use_beta_sigmoid_in_kernel,
                  allow_neg_eigval, safe_gate, state_v_first,
                  is_tnd, is_state_fp32,
                  static_cast<float>(scale), static_cast<float>(lower_bound),
                  aivNum, ubSize, tiling);

    // Set gate/beta dtype codes
    auto dtypeCode = [](at::ScalarType t) -> uint32_t {
        if (t == at::kFloat) return 0;
        if (t == at::kBFloat16) return 1;
        return 2;
    };
    tiling.gateDtype = dtypeCode(gate.scalar_type());
    tiling.betaDtype = dtypeCode(beta.scalar_type());

    if (has_accepted_tokens) {
        tiling.acceptedTokensDtype = (num_accepted_tokens_opt.value().scalar_type() == at::kInt) ? 0 : 1;
    }

    // Prepare output tensor (same shape as value)
    at::Tensor output = at::empty_like(value);

    // Prepare optional tensors (use empty tensors for missing optionals)
    auto empty_int = [&query]() {
        return at::empty({0}, query.options().dtype(at::kInt));
    };

    at::Tensor cu_seqlens_tensor = has_cu_seqlens
        ? cu_seqlens.contiguous()
        : empty_int();

    at::Tensor ssm_state_indices_tensor = has_ssm_state_indices
        ? ssm_state_indices.contiguous()
        : empty_int();

    at::Tensor a_log_tensor = has_a_log
        ? a_log_opt.value().contiguous()
        : at::empty({0}, query.options().dtype(at::kFloat));

    at::Tensor dt_bias_tensor = has_dt_bias
        ? dt_bias_opt.value().contiguous()
        : at::empty({0}, query.options().dtype(at::kFloat));

    at::Tensor num_accepted_tokens_tensor = has_accepted_tokens
        ? num_accepted_tokens_opt.value().to(at::kInt).contiguous()
        : empty_int();

    // Copy tiling data to device
    int32_t tilingSize =
        (static_cast<int32_t>(sizeof(RecurrentKdaTilingData)) + static_cast<int32_t>(PADDING_BYTE) - 1)
        / static_cast<int32_t>(PADDING_BYTE) * static_cast<int32_t>(PADDING_BYTE);

    auto cpuTiling = at::empty({tilingSize}, at::kByte);
    std::memcpy(cpuTiling.data_ptr(), &tiling, sizeof(RecurrentKdaTilingData));
    at::Tensor tilingTensor = TorchNpuHelper::CopyTensorHostToDevice(cpuTiling);

    // Allocate workspace
    auto workspaceTensor = at::empty({static_cast<int64_t>(RKDA_SYS_WORKSPACE_SIZE)},
                                     at::TensorOptions().dtype(at::kByte).device(query.options().device()));

    uint32_t blockDim = tiling.vectorCoreNum;
    if (blockDim == 0) {
        blockDim = 1;
    }

    at::Tensor query_c = query.contiguous();
    at::Tensor key_c = key.contiguous();
    at::Tensor value_c = value.contiguous();
    at::Tensor gate_c = gate.contiguous();
    at::Tensor beta_c = beta.contiguous();

    EXEC_KERNEL_CMD(recurrent_kda, blockDim,
                    query_c, key_c, value_c,
                    gate_c, beta_c,
                    initial_state,
                    cu_seqlens_tensor, ssm_state_indices_tensor,
                    a_log_tensor, dt_bias_tensor, num_accepted_tokens_tensor,
                    output, initial_state, initial_state,
                    workspaceTensor, tilingTensor);

    return output;
}

}  // namespace npu_kernel
}  // namespace sglang
