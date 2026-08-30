/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
 * the software repository for the full text of the License.
 */
#ifndef CATLASS_GEMM_KERNEL_GROUPED_MATMUL_SLICE_M_GELU_KERNEL_HPP
#define CATLASS_GEMM_KERNEL_GROUPED_MATMUL_SLICE_M_GELU_KERNEL_HPP

#include <iostream>

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/coord.hpp"
#include "catlass/detail/callback.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

namespace Catlass::Gemm::Kernel {

// Template for grouped matmul kernel. Compute grouped C = gelu(A * B)
template <class BlockMmadTla_, class BlockEpilogue_, class BlockScheduler_, class ElementGroupList_>
class GroupedMatmulSliceMGelu {
public:
    using BlockMmadTla = BlockMmadTla_;
    using BlockEpilogue = BlockEpilogue_;
    using ArchTag = typename BlockMmadTla::ArchTag;
    using L1TileShape = typename BlockMmadTla::L1TileShape;
    using L0TileShape = typename BlockMmadTla::L0TileShape;
    using ElementA = typename BlockMmadTla::ElementA;
    using LayoutA = typename BlockMmadTla::LayoutA;
    using ElementB = typename BlockMmadTla::ElementB;
    using LayoutB = typename BlockMmadTla::LayoutB;
    using ElementMmOut = typename BlockMmadTla::ElementC;
    using LayoutMmOut = typename BlockMmadTla::LayoutC;
    using LayoutTagMmOut = layout::RowMajor;
    using ElementAccumulator = typename BlockMmadTla::ElementAccumulator;

    using ElementO = typename BlockEpilogue::ElementDst;
    using LayoutTagO = typename BlockEpilogue::LayoutTagDst;

    using TensorMmOut = tla::Tensor<
        AscendC::LocalTensor<ElementMmOut>, LayoutMmOut, tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::VECCALC>;

    using ElementGroupList = ElementGroupList_;

    using BlockScheduler = BlockScheduler_;

    static constexpr uint32_t UB_STAGES = BlockEpilogue::UB_STAGES;

    static constexpr uint32_t L1_TILE_M = tla::get<0>(L1TileShape{});
    static constexpr uint32_t L1_TILE_N = tla::get<1>(L1TileShape{});
    static constexpr uint32_t L1_TILE_K = tla::get<2>(L1TileShape{});

    static constexpr uint32_t L0_TILE_M = tla::get<0>(L0TileShape{});
    static constexpr uint32_t L0_TILE_N = tla::get<1>(L0TileShape{});
    static constexpr uint32_t L0_TILE_K = tla::get<2>(L0TileShape{});

    static_assert(
        (sizeof(ElementMmOut) * L1_TILE_N) % BYTE_PER_BLK == 0, "ElementMmOut must be divisible by BYTE_PER_BLK");

    static constexpr uint32_t MM_OUT_SIZE_PING = RoundUp<256>(L1_TILE_M / 2 * L1_TILE_N * sizeof(ElementMmOut));
    static constexpr uint32_t MM_OUT_SIZE = MM_OUT_SIZE_PING * UB_STAGES;

    static_assert((MM_OUT_SIZE <= ArchTag::UB_SIZE), "MM_OUT_SIZE must be less than or equal to UB_SIZE");

    static constexpr uint32_t UB_OFFSET_MM_OUT = 0;

    struct Params {
        GemmCoord problemShape;
        uint32_t problemCount;
        __gm__ ElementGroupList* ptrGroupList;
        __gm__ ElementA* ptrA;
        LayoutA layoutA;
        __gm__ ElementB* ptrB;
        LayoutB layoutB;
        __gm__ ElementO* ptrO;

        CATLASS_HOST_DEVICE
        Params()
        {}

        CATLASS_HOST_DEVICE
        Params(
            GemmCoord const& problemShape_, uint32_t problemCount_, GM_ADDR ptrGroupList_, GM_ADDR ptrA_,
            LayoutA const& layoutA_, GM_ADDR ptrB_, LayoutB const& layoutB_, GM_ADDR ptrO_)
            : problemShape(problemShape_),
              problemCount(problemCount_),
              ptrGroupList(reinterpret_cast<__gm__ ElementGroupList*>(ptrGroupList_)),
              ptrA(reinterpret_cast<__gm__ ElementA*>(ptrA_)),
              layoutA(layoutA_),
              ptrB(reinterpret_cast<__gm__ ElementB*>(ptrB_)),
              layoutB(layoutB_),
              ptrO(reinterpret_cast<__gm__ ElementO*>(ptrO_))
        {}
    };

    struct Arguments {
        GemmCoord problemShape;
        uint32_t problemCount;
        uint8_t* ptrGroupList;
        uint8_t* ptrA;
        LayoutA layoutA;
        uint8_t* ptrB;
        LayoutB layoutB;
        uint8_t* ptrO;
    };

    static bool CanImplement(const Arguments& args)
    {
        if (args.problemShape.n() <= 0 || args.problemShape.k() <= 0) {
            std::cerr << "[ERROR]problemShape.n() or problemShape.k() must be > 0: "
                      << "n = " << args.problemShape.n() << ", k = " << args.problemShape.k() << std::endl;
            return false;
        }
        if (args.ptrGroupList == nullptr || args.ptrA == nullptr || args.ptrB == nullptr || args.ptrO == nullptr) {
            std::cerr << "[ERROR]ptrGroupList, ptrA, ptrB, ptrO cannot be nullptr: "
                      << "ptrGroupList = " << args.ptrGroupList << ", ptrA = " << args.ptrA << ", ptrB = " << args.ptrB
                      << ", ptrO = " << args.ptrO << std::endl;
            return false;
        }
        return true;
    }

    static size_t GetWorkspaceSize(const Arguments& args)
    {
        return 0;
    }

    static Params ToUnderlyingArguments(const Arguments& args, void* workspace)
    {
        Params params{args.problemShape, args.problemCount, args.ptrGroupList, args.ptrA,
                      args.layoutA,      args.ptrB,         args.layoutB,      args.ptrO};
        return params;
    }

    CATLASS_HOST_DEVICE
    GroupedMatmulSliceMGelu()
    {
        for (uint32_t ubStageId = 0; ubStageId < UB_STAGES; ++ubStageId) {
#ifdef __DAV_CUBE__
            mmadWaitUBBuffIdleFuncList[ubStageId] = MmadWaitUBBuffIdle{this, ubStageId};
            mmadSetUBBuffReadyFuncList[ubStageId] = MmadSetUBBuffReady{this, ubStageId};
#endif
#ifdef __DAV_VEC__
            GeluSetUBBuffIdleProc(ubStageId);
#endif
        }
    }

    CATLASS_HOST_DEVICE
    ~GroupedMatmulSliceMGelu()
    {
        for (uint32_t ubStageId = 0; ubStageId < UB_STAGES; ++ubStageId) {
#ifdef __DAV_CUBE__
            MmadWaitUBBuffIdleProc(ubStageId);
#endif
        }
    }

    CATLASS_DEVICE
    uint64_t CalcGmToL1BuffSize(uint64_t aSize, uint64_t bSize, uint32_t m, uint32_t n, uint32_t tileM, uint32_t tileN)
    {
        uint32_t mNum = (m + tileM - 1) / tileM;
        uint32_t nNum = (n + tileN - 1) / tileN;
        return (uint64_t)(aSize)*nNum + (uint64_t)(bSize)*mNum;
    }

    CATLASS_DEVICE
    GemmCoord GetL1TileShape(GemmCoord const& problemShape)
    {
        static constexpr uint16_t L1TileShape[][3] = {
            {128, 256, 256}, {192, 256, 256}, {240, 256, 256}, {256, 128, 256}, {256, 192, 256}};
        static constexpr uint16_t shapeNum = sizeof(L1TileShape) / sizeof(L1TileShape[0]);
        static_assert(shapeNum > 0, "empty tile candidate list");

        uint32_t m = problemShape.m();
        uint32_t n = problemShape.n();
        uint32_t k = problemShape.k();
        uint32_t shapeIdx = 0;
        uint64_t aSize = static_cast<uint64_t>(m) * k * sizeof(ElementA) / 1024; // unit KB
        uint64_t bSize = static_cast<uint64_t>(k) * n * sizeof(ElementB) / 1024; // unit KB
        uint64_t dmaSize = CalcGmToL1BuffSize(aSize, bSize, m, n, L1TileShape[0][0], L1TileShape[0][1]);

        for (uint32_t idx = 1; idx < shapeNum; ++idx) {
            uint64_t dmaSizeTmp = CalcGmToL1BuffSize(aSize, bSize, m, n, L1TileShape[idx][0], L1TileShape[idx][1]);
            if (dmaSizeTmp < dmaSize) {
                dmaSize = dmaSizeTmp;
                shapeIdx = idx;
            }
        }
        return GemmCoord{L1TileShape[shapeIdx][0], L1TileShape[shapeIdx][1], L1TileShape[shapeIdx][2]};
    }

    CATLASS_DEVICE
    void operator()(Params const& params)
    {
        uint32_t coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        uint32_t coreNum = AscendC::GetBlockNum();

        BlockScheduler blockScheduler;

        AscendC::GlobalTensor<ElementGroupList> groupList;
        groupList.SetGlobalBuffer(params.ptrGroupList);

        AscendC::LocalTensor<ElementMmOut> ubMmOut[UB_STAGES];
        for (uint32_t idx = 0; idx < UB_STAGES; ++idx) {
            ubMmOut[idx] =
                resource.ubBuf.template GetBufferByByte<ElementMmOut>(UB_OFFSET_MM_OUT + idx * MM_OUT_SIZE_PING);
        }

#ifdef __DAV_CUBE__
        BlockMmadTla blockMmadTla(resource);
        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer(params.ptrA);
        auto tensorA = tla::MakeTensor(gmA, params.layoutA, Arch::PositionGM{});
#endif

#ifdef __DAV_VEC__
        typename BlockEpilogue::Params epilogueParams{};
        BlockEpilogue blockEpilogue(resource, epilogueParams);
        AscendC::GlobalTensor<ElementO> gmO;
        gmO.SetGlobalBuffer(params.ptrO);
#endif

        uint32_t ubStageId = 0;
        int64_t groupGmBOffset = 0;
        int64_t groupMOffset = 0;
        uint32_t sumBlockNum = 0;
        uint32_t globalBlockIdx = coreIdx;

        for (uint32_t groupIdx = 0; groupIdx < params.problemCount; ++groupIdx) {
            uint32_t groupM = (groupIdx == 0) ? groupList.GetValue(groupIdx) :
                                                (groupList.GetValue(groupIdx) - groupList.GetValue(groupIdx - 1));

            GemmCoord groupProblemShape{groupM, params.problemShape.n(), params.problemShape.k()};

            GemmCoord l1TileShape = GetL1TileShape(groupProblemShape);
            GemmCoord l0TileShape{l1TileShape.m(), l1TileShape.n(), L0_TILE_K};

            uint32_t l1TileM = l1TileShape.m();
            uint32_t l1TileN = l1TileShape.n();
            uint32_t l1TileK = l1TileShape.k();
            uint32_t l0TileM = l0TileShape.m();
            uint32_t l0TileN = l0TileShape.n();
            uint32_t l0TileK = l0TileShape.k();

#ifdef __DAV_CUBE__
            AscendC::GlobalTensor<ElementB> gmB;
            gmB.SetGlobalBuffer(params.ptrB + groupGmBOffset);
            if (CeilDiv(groupM, l1TileM) == 1) {
                gmB.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
            }
            auto tensorB = tla::MakeTensor(gmB, params.layoutB, Arch::PositionGM{});
#endif
            blockScheduler.Update(groupProblemShape, MakeCoord(l1TileM, l1TileN));
            uint32_t coreLoops = blockScheduler.GetCoreLoops();

            for (; globalBlockIdx < sumBlockNum + coreLoops; globalBlockIdx += coreNum) {
                uint32_t localBlockIdx = globalBlockIdx - sumBlockNum;     // local block idx
                uint32_t loopIdx = (localBlockIdx + groupIdx) % coreLoops; // block idx remap

                GemmCoord blockCoord = blockScheduler.GetBlockCoord(loopIdx);
                GemmCoord actualBlockShape = blockScheduler.GetActualBlockShape(blockCoord);

                uint32_t blockColumnRound = RoundUp<16>(actualBlockShape.n()); // 32byte alignment
                LayoutTagMmOut layoutTagBlockMmOut{actualBlockShape.m(), blockColumnRound};
                auto layoutBlockMmOut = tla::MakeLayoutFromTag(layoutTagBlockMmOut);
                auto tensorBlockMmOut = tla::MakeTensor(ubMmOut[ubStageId], layoutBlockMmOut, Arch::PositionUB{});

#ifdef __DAV_CUBE__
                auto tensorBlockA = GetTile(
                    tensorA, tla::MakeCoord(groupMOffset + blockCoord.m() * l1TileM, blockCoord.k() * l1TileK),
                    tla::MakeShape(actualBlockShape.m(), actualBlockShape.k()));
                auto tensorBlockB = GetTile(
                    tensorB, tla::MakeCoord(blockCoord.k() * l1TileK, blockCoord.n() * l1TileN),
                    tla::MakeShape(actualBlockShape.k(), actualBlockShape.n()));
                Callback callbackBeforeFixpipe = MakeCallback(&mmadWaitUBBuffIdleFuncList[ubStageId]);
                Callback callbackAfterFixpipe = MakeCallback(&mmadSetUBBuffReadyFuncList[ubStageId]);

                if constexpr (BlockMmadTla::DispatchPolicy::ENABLE_DYNAMIC_TILE == false) {
                    if constexpr (BlockMmadTla::DispatchPolicy::ASYNC) {
                        blockMmadTla(
                            tensorBlockA, tensorBlockB, tensorBlockMmOut, actualBlockShape, EmptyClass(),
                            callbackBeforeFixpipe, callbackAfterFixpipe);
                    } else {
                        callbackBeforeFixpipe();
                        blockMmadTla(tensorBlockA, tensorBlockB, tensorBlockMmOut, actualBlockShape);
                        callbackAfterFixpipe();
                    }
                } else {
                    if constexpr (BlockMmadTla::DispatchPolicy::ASYNC) {
                        blockMmadTla.computeByDynamicTile(
                            tensorBlockA, tensorBlockB, tensorBlockMmOut, actualBlockShape, l1TileShape, l0TileShape,
                            EmptyClass(), callbackBeforeFixpipe, callbackAfterFixpipe);
                    } else {
                        callbackBeforeFixpipe();
                        blockMmadTla.computeByDynamicTile(
                            tensorBlockA, tensorBlockB, tensorBlockMmOut, actualBlockShape, l1TileShape, l0TileShape);
                        callbackAfterFixpipe();
                    }
                }
#endif
#ifdef __DAV_VEC__
                auto ubBlockMmOut = tensorBlockMmOut.data();

                AscendC::GlobalTensor<ElementO> groupGmO = gmO[groupMOffset * params.problemShape.n()];
                LayoutTagO layoutTagGroupO{groupProblemShape.m(), groupProblemShape.n()};
                auto layoutTagBlockO = layoutTagGroupO.GetTileLayout(actualBlockShape.GetCoordMN());

                GemmCoord tileShape{l1TileM, l1TileN, l1TileK};
                auto blockOffset = layoutTagGroupO.GetOffset(blockCoord.GetCoordMN() * tileShape.GetCoordMN());
                auto gmBlockO = groupGmO[blockOffset];

                GeluWaitUBBuffReadyProc(ubStageId);
                blockEpilogue(
                    gmBlockO, layoutTagBlockO, ubBlockMmOut, layoutTagBlockMmOut, actualBlockShape, ubStageId);
                GeluSetUBBuffIdleProc(ubStageId);
#endif
                ubStageId = (ubStageId + 1 < UB_STAGES) ? (ubStageId + 1) : 0;
            }
            groupMOffset += groupProblemShape.m();
            groupGmBOffset += static_cast<int64_t>(groupProblemShape.k()) * groupProblemShape.n();

            sumBlockNum += coreLoops;
        }
#ifdef __DAV_CUBE__
        if constexpr (BlockMmadTla::DispatchPolicy::ASYNC) {
            blockMmadTla.template SynchronizeBlock<TensorMmOut>();
        }
#endif
        AscendC::PipeBarrier<PIPE_ALL>();
    }

#ifdef __DAV_CUBE__
    struct MmadWaitUBBuffIdle {
        using Kernel = GroupedMatmulSliceMGelu;

        CATLASS_DEVICE
        MmadWaitUBBuffIdle() = default;

        CATLASS_DEVICE
        MmadWaitUBBuffIdle(Kernel* kernel, uint32_t ubStageId) : kernelPtr(kernel), ubStageId(ubStageId)
        {}

        CATLASS_DEVICE
        void operator()() const
        {
            kernelPtr->MmadWaitUBBuffIdleProc(ubStageId);
        }

        Kernel* kernelPtr{nullptr};
        uint32_t ubStageId{0};
    };

    struct MmadSetUBBuffReady {
        using Kernel = GroupedMatmulSliceMGelu;

        CATLASS_DEVICE
        MmadSetUBBuffReady() = default;

        CATLASS_DEVICE
        MmadSetUBBuffReady(Kernel* kernel, uint32_t ubStageId) : kernelPtr(kernel), ubStageId(ubStageId)
        {}

        CATLASS_DEVICE
        void operator()() const
        {
            kernelPtr->MmadSetUBBuffReadyProc(ubStageId);
        }

        Kernel* kernelPtr{nullptr};
        uint32_t ubStageId{0};
    };

    CATLASS_DEVICE void MmadWaitUBBuffIdleProc(uint16_t ubStageId)
    {
        AscendC::CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE_4, PIPE_FIX>(AIV_TO_AIC_FLAG_ID + ubStageId);
        AscendC::CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE_4, PIPE_FIX>(AIV_TO_AIC_FLAG_ID + FLAG_ID_MAX + ubStageId);
    }

    CATLASS_DEVICE void MmadSetUBBuffReadyProc(uint16_t ubStageId)
    {
        AscendC::CrossCoreSetFlag<CROSS_CORE_SYNC_MODE_4, PIPE_FIX>(AIC_TO_AIV_FLAG_ID + ubStageId);
        AscendC::CrossCoreSetFlag<CROSS_CORE_SYNC_MODE_4, PIPE_FIX>(AIC_TO_AIV_FLAG_ID + FLAG_ID_MAX + ubStageId);
    }
#endif

#ifdef __DAV_VEC__
    CATLASS_DEVICE void GeluWaitUBBuffReadyProc(uint16_t ubStageId)
    {
        AscendC::CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE_4, PIPE_V>(AIC_TO_AIV_FLAG_ID + ubStageId);
    }

    CATLASS_DEVICE void GeluSetUBBuffIdleProc(uint16_t ubStageId)
    {
        AscendC::CrossCoreSetFlag<CROSS_CORE_SYNC_MODE_4, PIPE_MTE3>(AIV_TO_AIC_FLAG_ID + ubStageId);
    }
#endif

private:
#ifdef __DAV_CUBE__
    MmadWaitUBBuffIdle mmadWaitUBBuffIdleFuncList[UB_STAGES];
    MmadSetUBBuffReady mmadSetUBBuffReadyFuncList[UB_STAGES];
#endif

    Arch::Resource<ArchTag> resource;

    static constexpr uint16_t CROSS_CORE_SYNC_MODE_4 = 4;

    static constexpr uint16_t AIV_TO_AIC_FLAG_ID = 6;
    static constexpr uint16_t AIC_TO_AIV_FLAG_ID = 8;
    static constexpr uint16_t FLAG_ID_MAX = 16;
};

} // namespace Catlass::Gemm::Kernel

#endif // CATLASS_GEMM_KERNEL_GROUPED_MATMUL_SLICE_M_GELU_KERNEL_HPP
