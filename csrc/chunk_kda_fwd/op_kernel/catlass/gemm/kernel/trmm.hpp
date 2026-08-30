/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_KERNEL_TRMM_HPP
#define CATLASS_GEMM_KERNEL_TRMM_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/coord.hpp"
#include "catlass/epilogue/tile/copy_gm_to_ub_tla.hpp"
#include "catlass/epilogue/tile/copy_ub_to_gm_tla.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

namespace Catlass::Gemm::Kernel {

// Compute TRMM C = alpha * op(T) * B or C = alpha * A * op(T), where T is triangular.
// Supported kernel parameters are side={left,right}, uplo={lower,upper}, trans={no,trans}, and diag={nonunit}.
// The kernel reduces work by clipping each output tile to the valid triangular K range and expects inactive triangular
// elements to be zeroed by the caller-side wrapper.
template <class BlockMmad_, class BlockEpilogue_, class BlockScheduler_>
class Trmm {
public:
    using BlockMmad = BlockMmad_;
    using ArchTag = typename BlockMmad::ArchTag;
    using L1TileShape = typename BlockMmad::L1TileShape;
    using ElementA = typename BlockMmad::ElementA;
    using LayoutA = typename BlockMmad::LayoutA;
    using ElementB = typename BlockMmad::ElementB;
    using LayoutB = typename BlockMmad::LayoutB;
    using ElementC = typename BlockMmad::ElementC;
    using LayoutC = typename BlockMmad::LayoutC;
    using ElementAccumulator = typename BlockMmad::ElementAccumulator;

    using BlockEpilogue = BlockEpilogue_;
    using BlockScheduler = BlockScheduler_;

    static constexpr uint32_t L1_TILE_M = tla::get<0>(L1TileShape{});
    static constexpr uint32_t L1_TILE_N = tla::get<1>(L1TileShape{});
    static constexpr uint32_t L1_TILE_K = tla::get<2>(L1TileShape{});

    struct Params {
        GemmCoord problemShape;
        GM_ADDR ptrA;
        LayoutA layoutA;
        GM_ADDR ptrB;
        LayoutB layoutB;
        GM_ADDR ptrC;
        LayoutC layoutC;
        uint32_t side;
        uint32_t uplo;
        uint32_t trans;
        uint32_t diag;
        float alpha;

        CATLASS_HOST_DEVICE
        Params()
        {}

        CATLASS_HOST_DEVICE
        Params(
            GemmCoord const& problemShape_, GM_ADDR ptrA_, LayoutA layoutA_, GM_ADDR ptrB_, LayoutB layoutB_,
            GM_ADDR ptrC_, LayoutC layoutC_, uint32_t side_, uint32_t uplo_, uint32_t trans_, uint32_t diag_,
            float alpha_)
            : problemShape(problemShape_),
              ptrA(ptrA_),
              layoutA(layoutA_),
              ptrB(ptrB_),
              layoutB(layoutB_),
              ptrC(ptrC_),
              layoutC(layoutC_),
              side(side_),
              uplo(uplo_),
              trans(trans_),
              diag(diag_),
              alpha(alpha_)
        {}
    };

    struct Arguments {
        GemmCoord problemShape;
        GM_ADDR ptrA;
        LayoutA layoutA;
        GM_ADDR ptrB;
        LayoutB layoutB;
        GM_ADDR ptrC;
        LayoutC layoutC;
        uint32_t side{0};
        uint32_t uplo{0};
        uint32_t trans{0};
        uint32_t diag{0};
        float alpha{1.0f};
    };

    static bool CanImplement(const Arguments& args)
    {
        if (args.ptrA == nullptr || args.ptrB == nullptr || args.ptrC == nullptr) {
            return false;
        }
        if (args.problemShape.m() == 0 || args.problemShape.n() == 0 || args.problemShape.k() == 0) {
            return false;
        }
        if (args.side > 1 || args.uplo > 1 || args.trans > 1 || args.diag != 0) {
            return false;
        }
        if (args.side == 0) {
            return args.problemShape.m() == args.problemShape.k();
        }
        return args.problemShape.n() == args.problemShape.k();
    }

    static size_t GetWorkspaceSize(const Arguments& args)
    {
        return 0;
    }

    static Params ToUnderlyingArguments(const Arguments& args, uint8_t* workspace)
    {
        Params params{args.problemShape, args.ptrA, args.layoutA, args.ptrB,  args.layoutB, args.ptrC,
                      args.layoutC,      args.side, args.uplo,    args.trans, args.diag,    args.alpha};
        return params;
    }

    CATLASS_DEVICE
    Trmm()
    {}

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE void operator()(Params const& params);

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIC>(Params const& params)
    {
        if constexpr (std::is_same_v<ElementA, float> && std::is_same_v<ElementB, float>) {
            AscendC::SetHF32Mode(false);
        }
        BlockScheduler matmulBlockScheduler(params.problemShape, MakeCoord(L1_TILE_M, L1_TILE_N));
        uint32_t coreLoops = matmulBlockScheduler.GetCoreLoops();

        Arch::Resource<ArchTag> resource;
        BlockMmad blockMmad(resource);

        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer((__gm__ ElementA*)params.ptrA);
        AscendC::GlobalTensor<ElementB> gmB;
        gmB.SetGlobalBuffer((__gm__ ElementB*)params.ptrB);
        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer((__gm__ ElementC*)params.ptrC);

        auto tensorA = tla::MakeTensor(gmA, params.layoutA, Arch::PositionGM{});
        auto tensorB = tla::MakeTensor(gmB, params.layoutB, Arch::PositionGM{});
        auto tensorC = tla::MakeTensor(gmC, params.layoutC, Arch::PositionGM{});

        for (uint32_t loopIdx = AscendC::GetBlockIdx(); loopIdx < coreLoops; loopIdx += AscendC::GetBlockNum()) {
            GemmCoord blockCoord = matmulBlockScheduler.GetBlockCoord(loopIdx);
            GemmCoord actualBlockShape = matmulBlockScheduler.GetActualBlockShape(blockCoord);

            uint32_t mStart = blockCoord.m() * L1_TILE_M;
            uint32_t nStart = blockCoord.n() * L1_TILE_N;
            uint32_t mEnd = mStart + actualBlockShape.m();
            uint32_t nEnd = nStart + actualBlockShape.n();
            uint32_t kStart = 0;
            uint32_t kEnd = params.problemShape.k();
            uint32_t effectiveUplo = params.uplo ^ params.trans;
            if (params.side == 0) {
                if (effectiveUplo == 0) {
                    kEnd = mEnd < kEnd ? mEnd : kEnd;
                } else {
                    kStart = mStart < kEnd ? mStart : kEnd;
                }
            } else {
                if (effectiveUplo == 0) {
                    kStart = nStart < kEnd ? nStart : kEnd;
                } else {
                    kEnd = nEnd < kEnd ? nEnd : kEnd;
                }
            }
            if (kStart >= kEnd) {
                continue;
            }
            actualBlockShape.k() = kEnd - kStart;

            auto tensorBlockA = GetTile(
                tensorA, tla::MakeCoord(mStart, kStart), tla::MakeShape(actualBlockShape.m(), actualBlockShape.k()));
            auto tensorBlockB = GetTile(
                tensorB, tla::MakeCoord(kStart, nStart), tla::MakeShape(actualBlockShape.k(), actualBlockShape.n()));
            auto tensorBlockC = GetTile(
                tensorC, tla::MakeCoord(mStart, nStart), tla::MakeShape(actualBlockShape.m(), actualBlockShape.n()));

            blockMmad(tensorBlockA, tensorBlockB, tensorBlockC, actualBlockShape);

            if (params.alpha != 1.0f) {
                Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>(flagAicFinishStore);
            }
        }

        AscendC::PipeBarrier<PIPE_ALL>();
    }

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIV>(Params const& params)
    {
        if (params.alpha == 1.0f) {
            return;
        }

        BlockScheduler matmulBlockScheduler(params.problemShape, MakeCoord(L1_TILE_M, L1_TILE_N));
        uint32_t coreLoops = matmulBlockScheduler.GetCoreLoops();

        Arch::Resource<ArchTag> resource;
        auto ubTensor = resource.ubBuf.template GetBufferByByte<ElementC>(0);

        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer((__gm__ ElementC*)params.ptrC);
        auto tensorC = tla::MakeTensor(gmC, params.layoutC, Arch::PositionGM{});

        uint32_t subBlockNum = AscendC::GetSubBlockNum();
        uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        uint32_t aicoreIndex = AscendC::GetBlockIdx() / subBlockNum;
        uint32_t aicoreNum = AscendC::GetBlockNum() / subBlockNum;

        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);

        for (uint32_t loopIdx = aicoreIndex; loopIdx < coreLoops; loopIdx += aicoreNum) {
            GemmCoord blockCoord = matmulBlockScheduler.GetBlockCoord(loopIdx);
            GemmCoord actualBlockShape = matmulBlockScheduler.GetActualBlockShape(blockCoord);
            uint32_t mStart = blockCoord.m() * L1_TILE_M;
            uint32_t nStart = blockCoord.n() * L1_TILE_N;
            uint32_t mEnd = mStart + actualBlockShape.m();
            uint32_t nEnd = nStart + actualBlockShape.n();
            uint32_t kStart = 0;
            uint32_t kEnd = params.problemShape.k();
            uint32_t effectiveUplo = params.uplo ^ params.trans;
            if (params.side == 0) {
                if (effectiveUplo == 0) {
                    kEnd = mEnd < kEnd ? mEnd : kEnd;
                } else {
                    kStart = mStart < kEnd ? mStart : kEnd;
                }
            } else {
                if (effectiveUplo == 0) {
                    kStart = nStart < kEnd ? nStart : kEnd;
                } else {
                    kEnd = nEnd < kEnd ? nEnd : kEnd;
                }
            }
            if (kStart >= kEnd) {
                continue;
            }
            Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE3>(flagAicFinishStore);

            uint32_t subBlockM = CeilDiv(actualBlockShape.m(), subBlockNum);
            uint32_t subBlockMOffset = subBlockIdx * subBlockM;
            if (subBlockMOffset >= actualBlockShape.m()) {
                continue;
            }
            uint32_t actualSubBlockM = actualBlockShape.m() - subBlockMOffset;
            actualSubBlockM = actualSubBlockM < subBlockM ? actualSubBlockM : subBlockM;

            auto tensorBlockC = GetTile(
                tensorC, tla::MakeCoord(mStart + subBlockMOffset, nStart),
                tla::MakeShape(actualSubBlockM, actualBlockShape.n()));
            auto layoutUbC = tla::MakeLayout(
                tla::MakeShape(actualSubBlockM, actualBlockShape.n()),
                tla::MakeStride(actualBlockShape.n(), tla::Int<1>{}));
            auto tensorUbC = tla::MakeTensor(ubTensor, layoutUbC, Arch::PositionUB{});

            using CopyGmToUbC = Epilogue::Tile::CopyGm2UbTla<ArchTag, decltype(tensorBlockC), decltype(tensorUbC)>;
            using CopyUbToGmC = Epilogue::Tile::CopyUb2GmTla<ArchTag, decltype(tensorUbC), decltype(tensorBlockC)>;
            CopyGmToUbC copyGmToUbC;
            CopyUbToGmC copyUbToGmC;

            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            copyGmToUbC(tensorUbC, tensorBlockC);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::Muls(
                tensorUbC.data(), tensorUbC.data(), static_cast<ElementC>(params.alpha),
                actualSubBlockM * actualBlockShape.n());
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);

            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            copyUbToGmC(tensorBlockC, tensorUbC);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        }

        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

private:
    static constexpr Arch::FlagID FLAG_AIC_FINISH_STORE = 0;
    static constexpr Arch::FlagID RV_FLAG_AIC_FINISH_STORE = 1;
    Arch::CrossCoreFlagWithReverse<> flagAicFinishStore{FLAG_AIC_FINISH_STORE, RV_FLAG_AIC_FINISH_STORE};
};

} // namespace Catlass::Gemm::Kernel

#endif // CATLASS_GEMM_KERNEL_TRMM_HPP
