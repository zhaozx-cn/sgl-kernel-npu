/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
 */

#ifndef SGL_KERNEL_NPU_RECURRENT_KDA_HOST_H_
#define SGL_KERNEL_NPU_RECURRENT_KDA_HOST_H_

#include <ATen/ATen.h>
#include "defines.h"

namespace sglang {
namespace npu_kernel {

HOST_API at::Tensor recurrent_kda_impl(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
    const at::Tensor &gate, const at::Tensor &beta, at::Tensor &initial_state,
    const at::Tensor &cu_seqlens, const at::Tensor &ssm_state_indices,
    const c10::optional<at::Tensor> &a_log_opt, const c10::optional<at::Tensor> &dt_bias_opt,
    const c10::optional<at::Tensor> &num_accepted_tokens_opt,
    double scale, bool use_qk_l2norm_in_kernel, bool use_gate_in_kernel,
    bool use_beta_sigmoid_in_kernel, bool allow_neg_eigval,
    bool safe_gate, double lower_bound, bool state_v_first);

}  // namespace npu_kernel
}  // namespace sglang

#endif  // SGL_KERNEL_NPU_RECURRENT_KDA_HOST_H_
