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
 * \brief direct-launch entry of chunk_kda_fwd
 *
 * Mirrors the aclnn op_kernel entry, but replaces GET_TILING_DATA_WITH_STRUCT
 * (unavailable under direct-launch compilation) with a GM->stack tiling copy,
 * and replaces TILING_KEY_IS dispatch (no TilingContext under direct-launch)
 * with dispatch on the tiling struct fields: the 64/128/128 template shape is
 * re-derived from chunkSize/kHeadDim/vHeadDim, and the q/beta dtypes are
 * carried in qDataType/betaDataType (2=float, 1=bf16, 0=fp16).
 */

#include "kernel_operator.h"

#include "chunk_kda_fwd_common.h"
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#define KDA_COMPILE_ARCH35_FAST_PATH 1
#include "arch35/chunk_kda_fwd_impl.h"
#else
#define KDA_COMPILE_ARCH35_FAST_PATH 0
#endif

#include "chunk_kda_fwd_tiling_data.h"

namespace KdaForward {

constexpr int64_t KDA_DTYPE_FP16 = 0;
constexpr int64_t KDA_DTYPE_BF16 = 1;
constexpr int64_t KDA_DTYPE_FP32 = 2;

template <bool SAFE_GATE, typename BETA_T, typename TilingData,
          uint32_t COMPILE_BT, uint32_t COMPILE_K, uint32_t COMPILE_V>
__aicore__ inline void DispatchGeneric(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR g, GM_ADDR beta,
    GM_ADDR aLog, GM_ADDR dtBias, GM_ADDR initialState,
    GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR attnOut,
    GM_ADDR finalState, GM_ADDR gk, GM_ADDR aqk, GM_ADDR akk,
    GM_ADDR w, GM_ADDR u, GM_ADDR qg, GM_ADDR kg, GM_ADDR vNew, GM_ADDR h,
    GM_ADDR userWorkspace, const TilingData &tiling)
{
    RunGeneric<SAFE_GATE, bfloat16_t, BETA_T, TilingData,
               COMPILE_BT, COMPILE_K, COMPILE_V>(
        q, k, v, g, beta, aLog, dtBias, initialState, cuSeqlens,
        chunkIndices, attnOut, finalState, gk, aqk, akk, w, u, qg, kg,
        vNew, h, userWorkspace, tiling);
}

template <typename BETA_T, typename TilingData,
          uint32_t COMPILE_BT, uint32_t COMPILE_K, uint32_t COMPILE_V>
__aicore__ inline void DispatchGenericSafeGate(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR g, GM_ADDR beta,
    GM_ADDR aLog, GM_ADDR dtBias, GM_ADDR initialState,
    GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR attnOut,
    GM_ADDR finalState, GM_ADDR gk, GM_ADDR aqk, GM_ADDR akk,
    GM_ADDR w, GM_ADDR u, GM_ADDR qg, GM_ADDR kg, GM_ADDR vNew, GM_ADDR h,
    GM_ADDR userWorkspace, const TilingData &tiling)
{
    if (tiling.safeGate) {
        DispatchGeneric<true, BETA_T, TilingData,
                        COMPILE_BT, COMPILE_K, COMPILE_V>(
            q, k, v, g, beta, aLog, dtBias, initialState, cuSeqlens,
            chunkIndices, attnOut, finalState, gk, aqk, akk, w, u, qg,
            kg, vNew, h, userWorkspace, tiling);
    } else {
        DispatchGeneric<false, BETA_T, TilingData,
                        COMPILE_BT, COMPILE_K, COMPILE_V>(
            q, k, v, g, beta, aLog, dtBias, initialState, cuSeqlens,
            chunkIndices, attnOut, finalState, gk, aqk, akk, w, u, qg,
            kg, vNew, h, userWorkspace, tiling);
    }
}

#if KDA_COMPILE_ARCH35_FAST_PATH
template <bool SAFE_GATE, typename BETA_T, typename TilingData>
__aicore__ inline void DispatchArch35(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR g, GM_ADDR beta,
    GM_ADDR aLog, GM_ADDR dtBias, GM_ADDR initialState,
    GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR attnOut,
    GM_ADDR finalState, GM_ADDR gk, GM_ADDR aqk, GM_ADDR akk,
    GM_ADDR w, GM_ADDR u, GM_ADDR qg, GM_ADDR kg, GM_ADDR vNew, GM_ADDR h,
    GM_ADDR userWorkspace, const TilingData &tiling)
{
    AscendC::TPipe pipe;
    arch35::Run<SAFE_GATE, bfloat16_t, BETA_T, TilingData, 64, 128, 128>(
        q, k, v, g, beta, aLog, dtBias, initialState, cuSeqlens,
        chunkIndices, attnOut, finalState, gk, aqk, akk, w, u, qg, kg,
        vNew, h, userWorkspace, tiling, pipe);
}

template <typename BETA_T, typename TilingData>
__aicore__ inline void DispatchArch35SafeGate(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR g, GM_ADDR beta,
    GM_ADDR aLog, GM_ADDR dtBias, GM_ADDR initialState,
    GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR attnOut,
    GM_ADDR finalState, GM_ADDR gk, GM_ADDR aqk, GM_ADDR akk,
    GM_ADDR w, GM_ADDR u, GM_ADDR qg, GM_ADDR kg, GM_ADDR vNew, GM_ADDR h,
    GM_ADDR userWorkspace, const TilingData &tiling)
{
    if (tiling.safeGate) {
        DispatchArch35<true, BETA_T, TilingData>(
            q, k, v, g, beta, aLog, dtBias, initialState, cuSeqlens,
            chunkIndices, attnOut, finalState, gk, aqk, akk, w, u, qg,
            kg, vNew, h, userWorkspace, tiling);
    } else {
        DispatchArch35<false, BETA_T, TilingData>(
            q, k, v, g, beta, aLog, dtBias, initialState, cuSeqlens,
            chunkIndices, attnOut, finalState, gk, aqk, akk, w, u, qg,
            kg, vNew, h, userWorkspace, tiling);
    }
}

template <typename BETA_T, typename TilingData>
__aicore__ inline void Dispatch(
    bool useTemplate,
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR g, GM_ADDR beta,
    GM_ADDR aLog, GM_ADDR dtBias, GM_ADDR initialState,
    GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR attnOut,
    GM_ADDR finalState, GM_ADDR gk, GM_ADDR aqk, GM_ADDR akk,
    GM_ADDR w, GM_ADDR u, GM_ADDR qg, GM_ADDR kg, GM_ADDR vNew, GM_ADDR h,
    GM_ADDR userWorkspace, const TilingData &tiling)
{
    if (useTemplate) {
        DispatchArch35SafeGate<BETA_T, TilingData>(
            q, k, v, g, beta, aLog, dtBias, initialState, cuSeqlens,
            chunkIndices, attnOut, finalState, gk, aqk, akk, w, u, qg,
            kg, vNew, h, userWorkspace, tiling);
    } else {
        DispatchGenericSafeGate<BETA_T, TilingData, 0, 0, 0>(
            q, k, v, g, beta, aLog, dtBias, initialState, cuSeqlens,
            chunkIndices, attnOut, finalState, gk, aqk, akk, w, u, qg,
            kg, vNew, h, userWorkspace, tiling);
    }
}
#else
template <typename BETA_T, typename TilingData>
__aicore__ inline void Dispatch(
    bool,
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR g, GM_ADDR beta,
    GM_ADDR aLog, GM_ADDR dtBias, GM_ADDR initialState,
    GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR attnOut,
    GM_ADDR finalState, GM_ADDR gk, GM_ADDR aqk, GM_ADDR akk,
    GM_ADDR w, GM_ADDR u, GM_ADDR qg, GM_ADDR kg, GM_ADDR vNew, GM_ADDR h,
    GM_ADDR userWorkspace, const TilingData &tiling)
{
    // COMPILE_BT/K/V are inert on arch22, so the 64/128/128 template variant
    // shares the generic instantiation.
    DispatchGenericSafeGate<BETA_T, TilingData, 0, 0, 0>(
        q, k, v, g, beta, aLog, dtBias, initialState, cuSeqlens,
        chunkIndices, attnOut, finalState, gk, aqk, akk, w, u, qg,
        kg, vNew, h, userWorkspace, tiling);
}
#endif
}  // namespace KdaForward

extern "C" __global__ __aicore__ void chunk_kda_fwd(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR g, GM_ADDR beta,
    GM_ADDR a_log, GM_ADDR dt_bias, GM_ADDR initial_state,
    GM_ADDR cu_seqlens, GM_ADDR chunk_indices, GM_ADDR attn_out,
    GM_ADDR final_state, GM_ADDR gk, GM_ADDR aqk, GM_ADDR akk,
    GM_ADDR w, GM_ADDR u, GM_ADDR qg, GM_ADDR kg, GM_ADDR v_new, GM_ADDR h,
    GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(KdaForward::ChunkKdaFwdTilingData);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    GM_ADDR userWorkspace = AscendC::GetUserWorkspace(workspace);

    constexpr size_t TILING_WORDS = (sizeof(KdaForward::ChunkKdaFwdTilingData) + 7) / 8;
    alignas(8) uint64_t tilingBuf[TILING_WORDS];
    const __gm__ uint64_t *gmTilingWords = reinterpret_cast<const __gm__ uint64_t *>(tiling);
    for (size_t i = 0; i < TILING_WORDS; ++i) {
        tilingBuf[i] = gmTilingWords[i];
    }
    KdaForward::ChunkKdaFwdTilingData *tilingData =
        reinterpret_cast<KdaForward::ChunkKdaFwdTilingData *>(tilingBuf);

#if KDA_COMPILE_ARCH35_FAST_PATH
    const bool useTemplate = tilingData->chunkSize == 64 &&
                             tilingData->kHeadDim == 128 && tilingData->vHeadDim == 128;
#else
    constexpr bool useTemplate = false;
#endif

    if (tilingData->betaDataType == KdaForward::KDA_DTYPE_FP32) {
        KdaForward::Dispatch<float>(
            useTemplate, q, k, v, g, beta, a_log, dt_bias, initial_state,
            cu_seqlens, chunk_indices, attn_out, final_state, gk, aqk, akk,
            w, u, qg, kg, v_new, h, userWorkspace, *tilingData);
    } else {
        KdaForward::Dispatch<bfloat16_t>(
            useTemplate, q, k, v, g, beta, a_log, dt_bias, initial_state,
            cu_seqlens, chunk_indices, attn_out, final_state, gk, aqk, akk,
            w, u, qg, kg, v_new, h, userWorkspace, *tilingData);
    }
}

#undef KDA_COMPILE_ARCH35_FAST_PATH
