#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#include "arch35/recurrent_kda.h"
#else
#include "recurrent_kda.h"
#endif
#include "recurrent_kda_tiling_data.h"

using namespace AscendC;
using namespace matmul;
using namespace RecurrentKda;

extern "C" __global__ __aicore__ void
recurrent_kda(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR gate, GM_ADDR beta, GM_ADDR initialState,
              GM_ADDR cuSeqlens, GM_ADDR ssmStateIndices, GM_ADDR aLog, GM_ADDR dtBias, GM_ADDR numAcceptedTokens,
              GM_ADDR out, GM_ADDR initialStateOut, GM_ADDR finalState, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(RecurrentKdaTilingData);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    uint32_t tilingSize = (sizeof(RecurrentKdaTilingData) + 31) / 32 * 32;
    TBuf<TPosition::VECCALC> tilingBuf;
    pipe.InitBuffer(tilingBuf, tilingSize);
    auto localTiling = tilingBuf.Get<uint8_t>();
    GlobalTensor<uint8_t> gmTiling;
    gmTiling.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(tilingGM));
    DataCopy(localTiling, gmTiling, tilingSize);
    auto td = reinterpret_cast<RecurrentKdaTilingData *>(localTiling.GetPhyAddr());
    RKDA<bfloat16_t, bfloat16_t, float> op(td);
    GM_ADDR stateOutput = td->inplaceFinalState == 1 ? initialStateOut : finalState;
    RKDAInitParams initParams{query, key, value, gate, beta, initialState, cuSeqlens, ssmStateIndices,
                              aLog, dtBias, numAcceptedTokens, out, stateOutput};
    op.Init(initParams, &pipe);
    op.Process();
}