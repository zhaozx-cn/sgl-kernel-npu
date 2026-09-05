#ifndef ACLRTLAUNCH_RECURRENT_KDA_H
#define ACLRTLAUNCH_RECURRENT_KDA_H
#include "acl/acl_base.h"

#ifndef ACLRT_LAUNCH_KERNEL
#define ACLRT_LAUNCH_KERNEL(kernel_func) aclrtlaunch_##kernel_func
#endif

extern "C" uint32_t aclrtlaunch_recurrent_kda(uint32_t numBlocks, aclrtStream stream,
    void *query, void *key, void *value, void *gate, void *beta,
    void *initialState, void *cuSeqlens, void *ssmStateIndices,
    void *aLog, void *dtBias, void *numAcceptedTokens,
    void *out, void *initialStateOut, void *finalState,
    void *workspace, void *tiling);
#endif