/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_KERNEL_PLANAR_COMPLEX_GEMM_TLA_HPP
#define CATLASS_GEMM_KERNEL_PLANAR_COMPLEX_GEMM_TLA_HPP

#include <type_traits>

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/coord.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemv/tile/vec_copy_gm_to_ub.hpp"
#include "catlass/gemv/tile/vec_copy_ub_to_gm.hpp"
#include "catlass/layout/layout.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

namespace Catlass::Gemm::Kernel {

namespace detail {

/// Normalize the type aliases exposed by the two BlockMmad variants so that
/// the kernel below can refer to a single set of names regardless of which
/// path was selected at compile time.
///
/// - The four-pass variant uses a generic BlockMmadTla (dispatched via
///   MmadPingpong), a standard TLA-style BlockMmad that knows nothing about
///   real/imag — both halves reuse the same B layout and C layout. The
///   layout tag is surfaced via TileCopy::LayoutTagA/B/C. The 4-pass
///   orchestration (atomic-add cross terms) is implemented in the kernel
///   layer below.
/// - The fused variant (BlockMmadTla specialized for the
///   MmadPlanarComplexFused policy) is TLA-style:
///   it exposes LayoutTagA/B/C (the tags) for Params storage and LayoutA/B/C
///   (TLA concrete layouts) for internal use. We store the tag in Params so
///   both variants share the same Params layout type, and create TLA
///   layouts on the fly in the AIC path via tla::MakeLayout.
template <bool USE_FOUR_PASS, class BlockMmadFourPass, class BlockMmadFused>
struct PlanarTypeTraits {
    static_assert(
        DEPENDENT_FALSE<BlockMmadFourPass, BlockMmadFused>,
        "PlanarTypeTraits: only USE_FOUR_PASS = true/false are supported");
};

template <class BlockMmadFourPass, class BlockMmadFused>
struct PlanarTypeTraits</*USE_FOUR_PASS=*/true, BlockMmadFourPass, BlockMmadFused> {
    using BlockMmad = BlockMmadFourPass;
    using ArchTag = typename BlockMmad::ArchTag;
    using L1TileShape = typename BlockMmad::L1TileShape;
    using ElementA = typename BlockMmad::ElementA;
    using ElementB = typename BlockMmad::ElementB;
    using ElementCReal = typename BlockMmad::ElementC;
    using ElementCImag = typename BlockMmad::ElementC;
    using LayoutA = typename BlockMmad::TileCopy::LayoutTagA;
    using LayoutBReal = typename BlockMmad::TileCopy::LayoutTagB;
    using LayoutBImag = typename BlockMmad::TileCopy::LayoutTagB;
    using LayoutC = typename BlockMmad::TileCopy::LayoutTagC;
    using ElementAccumulator = typename BlockMmad::ElementAccumulator;
};

template <class BlockMmadFourPass, class BlockMmadFused>
struct PlanarTypeTraits</*USE_FOUR_PASS=*/false, BlockMmadFourPass, BlockMmadFused> {
    using BlockMmad = BlockMmadFused;
    using ArchTag = typename BlockMmad::ArchTag;
    using L1TileShape = typename BlockMmad::L1TileShape;
    using ElementA = typename BlockMmad::ElementA;
    using ElementB = typename BlockMmad::ElementB;
    using ElementCReal = typename BlockMmad::ElementCReal;
    using ElementCImag = typename BlockMmad::ElementCImag;
    using LayoutA = typename BlockMmad::LayoutTagA;
    using LayoutBReal = typename BlockMmad::LayoutTagB;
    using LayoutBImag = typename BlockMmad::LayoutTagB;
    using LayoutC = typename BlockMmad::LayoutTagC;
    using ElementAccumulator = typename BlockMmad::ElementAccumulator;
};

} // namespace detail

/// NegateMatrixAiv — AIV preprocessing component that negates a GM tensor
/// element-wise (dst = -src) into a GM workspace. Following the padding_matmul
/// pattern: the component is defined in the kernel header, takes Resource in
/// its constructor, and does its own per-core tiling internally so the AIV
/// path just calls operator()(gmDst, gmSrc, totalElements).
///
/// Single-buffered: src and dst each occupy half of UB. The compute (Muls) is
/// negligible vs GM bandwidth, so double-buffering adds overhead without benefit.
template <class ArchTag_, class Element_>
struct NegateMatrixAiv {
    using ArchTag = ArchTag_;
    using Element = Element_;

    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);

    // Each buffer occupies half of UB
    static constexpr uint32_t SRC_BUF_BYTES = ArchTag::UB_SIZE / 2;
    static constexpr uint32_t DST_BUF_BYTES = ArchTag::UB_SIZE / 2;
    static constexpr uint32_t SRC_BUF_ELEMS = SRC_BUF_BYTES / sizeof(Element);
    static constexpr uint32_t TILE_LENGTH = SRC_BUF_ELEMS;
    static constexpr uint32_t TILE_LENGTH_ALIGNED = RoundUp(TILE_LENGTH, ELE_NUM_PER_C0);

    using VType = Gemm::GemmType<Element, layout::VectorLayout>;
    using CopyGm2Ub = Gemv::Tile::VecCopyGmToUB<ArchTag, VType>;
    using CopyUb2Gm = Gemv::Tile::VecCopyUBToGm<ArchTag, VType>;

    CATLASS_DEVICE
    NegateMatrixAiv()
    {}

    CATLASS_DEVICE
    NegateMatrixAiv(Arch::Resource<ArchTag>& resource, uint32_t ubAddrStart = 0)
    {
        srcBuf = resource.ubBuf.template GetBufferByByte<Element>(ubAddrStart);
        dstBuf = resource.ubBuf.template GetBufferByByte<Element>(ubAddrStart + DST_BUF_BYTES);

        // Pre-set event flags so the first WaitFlag in the loop won't block.
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((event_t)(eventIdMte2));
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>((event_t)(eventIdMte2));
    }

    CATLASS_DEVICE
    ~NegateMatrixAiv()
    {
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(eventIdMte2));
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>((event_t)(eventIdMte2));
    }

    CATLASS_DEVICE
    void operator()(
        AscendC::GlobalTensor<Element> const& gmDst, AscendC::GlobalTensor<Element> const& gmSrc,
        uint32_t totalElements)
    {
        uint32_t aivNum = AscendC::GetBlockNum();
        uint32_t coreId = AscendC::GetBlockIdx();
        uint32_t totalTiles = CeilDiv(totalElements, TILE_LENGTH_ALIGNED);
        uint32_t tilesPerCore = CeilDiv(totalTiles, aivNum);
        uint32_t startTile = coreId * tilesPerCore;
        uint32_t endTile = min(startTile + tilesPerCore, totalTiles);

        for (uint32_t tileIdx = startTile; tileIdx < endTile; tileIdx++) {
            uint32_t curOffset = tileIdx * TILE_LENGTH_ALIGNED;
            uint32_t curLen = (tileIdx == totalTiles - 1) ? (totalElements - curOffset) : TILE_LENGTH_ALIGNED;

            // Wait for previous iteration's V and MTE3 to finish (buffer reuse safety)
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(eventIdMte2));
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>((event_t)(eventIdMte2));

            // GM → UB (MTE2 pipeline)
            copyGm2Ub(srcBuf, gmSrc[curOffset], curLen);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(eventIdV));

            // Wait for MTE2 → V: DataCopy must complete before Muls reads srcBuf
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((event_t)(eventIdV));

            // Compute: Muls(-1) (V pipeline)
            AscendC::SetMaskCount();
            AscendC::SetVectorMask<Element, AscendC::MaskMode::COUNTER>(curLen);
            AscendC::Muls<Element, false>(
                dstBuf, srcBuf, static_cast<Element>(-1), AscendC::MASK_PLACEHOLDER, 1, AscendC::UnaryRepeatParams{});
            AscendC::SetMaskNorm();
            AscendC::ResetMask();
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((event_t)(eventIdMte2));
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>((event_t)(eventIdMte3));

            // Wait for V → MTE3: Muls must complete before DataCopyPad reads dstBuf
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>((event_t)(eventIdMte3));

            // UB → GM (MTE3 pipeline)
            layout::VectorLayout dstLayout{curLen};
            layout::VectorLayout srcLayout{curLen};
            copyUb2Gm(gmDst[curOffset], dstBuf, dstLayout, srcLayout);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>((event_t)(eventIdMte2));
        }
    }

protected:
    AscendC::LocalTensor<Element> srcBuf;
    AscendC::LocalTensor<Element> dstBuf;

    CopyGm2Ub copyGm2Ub;
    CopyUb2Gm copyUb2Gm;

    static constexpr int32_t eventIdMte2 = 0;
    static constexpr int32_t eventIdV = 1;
    static constexpr int32_t eventIdMte3 = 2;
};

/// PlanarComplexGemm — unified kernel template covering both Planar Complex
/// algorithms. The variant is picked at compile time via USE_FOUR_PASS_ and
/// only one of `BlockMmadFourPass_` / `BlockMmadFused_` needs to be a real
/// type; the other may be `void` and will not be instantiated.
///
/// USE_FOUR_PASS_ = true  (selected by host when K is large and per-core tile
///                         count is high enough to amortize 4 pipeline drains):
///   4 sequential BlockMmad invocations per block, fixpipe atomic-add fuses
///   the imag-cross terms back onto C:
///     pass1: C_real  = A_real * B_real           (no atomic)
///     pass2: C_real += signed imaginary cross term (atomic add)
///     pass3: C_imag  = A_imag * B_real           (no atomic)
///     pass4: C_imag += A_real * B_imag           (atomic add)
///
/// USE_FOUR_PASS_ = false (default — cheaper when K is small or per-core tile
///                         count is too low to amortize drains):
///   Single fused K-loop with alternating A/B operands and a shared L0C
///   buffer (BlockMmadTla specialized for MmadPlanarComplexFused):
///     C_real = A_real * B_real + signed imaginary cross term (2K sub-iterations)
///     C_imag = A_imag * B_real + A_real * B_imag      (2K sub-iterations)
///   FixPipe of C_real overlaps with C_imag computation.
///
/// In both variants the negated imaginary operand is produced by the AIV Mix prologue into workspace.
template <bool USE_FOUR_PASS_, bool NEGATE_A_, class BlockMmadFourPass_, class BlockMmadFused_, class BlockScheduler_>
class PlanarComplexGemm {
public:
    static constexpr bool USE_FOUR_PASS = USE_FOUR_PASS_;
    static constexpr bool NEGATE_A = NEGATE_A_;

private:
    using Traits = detail::PlanarTypeTraits<USE_FOUR_PASS, BlockMmadFourPass_, BlockMmadFused_>;

public:
    using BlockMmad = typename Traits::BlockMmad;
    using ArchTag = typename Traits::ArchTag;
    using L1TileShape = typename Traits::L1TileShape;
    using ElementA = typename Traits::ElementA;
    using ElementB = typename Traits::ElementB;
    using ElementCReal = typename Traits::ElementCReal;
    using ElementCImag = typename Traits::ElementCImag;
    using LayoutA = typename Traits::LayoutA;
    using LayoutBReal = typename Traits::LayoutBReal;
    using LayoutBImag = typename Traits::LayoutBImag;
    using LayoutC = typename Traits::LayoutC;

    using BlockScheduler = BlockScheduler_;

    /// Host-side arguments (identical for both variants).
    struct Arguments {
        GemmCoord problemShape;
        GM_ADDR ptrAReal;
        GM_ADDR ptrAImag;
        GM_ADDR ptrAImagSigned;
        GM_ADDR ptrBReal;
        GM_ADDR ptrBImag;
        GM_ADDR ptrBImagSigned;
        GM_ADDR ptrCReal;
        GM_ADDR ptrCImag;
    };

    /// Device-side params.
    struct Params {
        GemmCoord problemShape;
        GM_ADDR ptrAReal;
        GM_ADDR ptrAImag;
        GM_ADDR ptrAImagSigned; // -A_imag workspace when NEGATE_A, otherwise A_imag
        GM_ADDR ptrBReal;
        GM_ADDR ptrBImag;       // original
        GM_ADDR ptrBImagSigned; // -B_imag workspace when !NEGATE_A, otherwise B_imag
        GM_ADDR ptrCReal;
        GM_ADDR ptrCImag;
        LayoutA layoutA;
        LayoutBReal layoutBReal;
        LayoutBImag layoutBImag;
        LayoutC layoutC;

        CATLASS_HOST_DEVICE
        Params()
        {}

        CATLASS_HOST_DEVICE
        Params(
            GemmCoord const& problemShape_, GM_ADDR ptrAReal_, GM_ADDR ptrAImag_, GM_ADDR ptrAImagSigned_,
            GM_ADDR ptrBReal_, GM_ADDR ptrBImag_, GM_ADDR ptrBImagSigned_, GM_ADDR ptrCReal_, GM_ADDR ptrCImag_,
            LayoutA layoutA_, LayoutBReal layoutBReal_, LayoutBImag layoutBImag_, LayoutC layoutC_)
            : problemShape(problemShape_),
              ptrAReal(ptrAReal_),
              ptrAImag(ptrAImag_),
              ptrAImagSigned(ptrAImagSigned_),
              ptrBReal(ptrBReal_),
              ptrBImag(ptrBImag_),
              ptrBImagSigned(ptrBImagSigned_),
              ptrCReal(ptrCReal_),
              ptrCImag(ptrCImag_),
              layoutA(layoutA_),
              layoutBReal(layoutBReal_),
              layoutBImag(layoutBImag_),
              layoutC(layoutC_)
        {}
    };

    static bool CanImplement(const Arguments& args)
    {
        return true;
    }

    static size_t GetWorkspaceSize(const Arguments& args)
    {
        if constexpr (NEGATE_A) {
            LayoutA layoutA = LayoutA::template MakeLayout<ElementA>(args.problemShape.m(), args.problemShape.k());
            return layoutA.Capacity() * sizeof(ElementA);
        } else {
            LayoutBImag layoutBImag =
                LayoutBImag::template MakeLayout<ElementB>(args.problemShape.k(), args.problemShape.n());
            return layoutBImag.Capacity() * sizeof(ElementB);
        }
    }

    static Params ToUnderlyingArguments(const Arguments& args, uint8_t* workspace)
    {
        LayoutA layoutA = LayoutA::template MakeLayout<ElementA>(args.problemShape.m(), args.problemShape.k());
        LayoutBReal layoutBReal =
            LayoutBReal::template MakeLayout<ElementB>(args.problemShape.k(), args.problemShape.n());
        LayoutBImag layoutBImag =
            LayoutBImag::template MakeLayout<ElementB>(args.problemShape.k(), args.problemShape.n());
        LayoutC layoutC = LayoutC::template MakeLayout<ElementCReal>(args.problemShape.m(), args.problemShape.n());
        GM_ADDR ptrAImagSigned = NEGATE_A ? workspace : args.ptrAImag;
        GM_ADDR ptrBImagSigned = NEGATE_A ? args.ptrBImag : workspace;
        return Params{args.problemShape, args.ptrAReal,  args.ptrAImag, ptrAImagSigned, args.ptrBReal,
                      args.ptrBImag,     ptrBImagSigned, args.ptrCReal, args.ptrCImag,  layoutA,
                      layoutBReal,       layoutBImag,    layoutC};
    }

    CATLASS_DEVICE
    PlanarComplexGemm()
    {}

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE void operator()(Params const& params);

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIC>(Params const& params)
    {
        Catlass::Arch::CrossCoreWaitFlag(flagAivFinishNegate);

        Arch::Resource<ArchTag> resource;

        AscendC::GlobalTensor<ElementA> gmAReal;
        gmAReal.SetGlobalBuffer((__gm__ ElementA*)params.ptrAReal);
        AscendC::GlobalTensor<ElementA> gmAImag;
        gmAImag.SetGlobalBuffer((__gm__ ElementA*)params.ptrAImag);
        AscendC::GlobalTensor<ElementA> gmAImagSigned;
        gmAImagSigned.SetGlobalBuffer((__gm__ ElementA*)params.ptrAImagSigned);
        AscendC::GlobalTensor<ElementB> gmBReal;
        gmBReal.SetGlobalBuffer((__gm__ ElementB*)params.ptrBReal);
        AscendC::GlobalTensor<ElementB> gmBImag;
        gmBImag.SetGlobalBuffer((__gm__ ElementB*)params.ptrBImag);
        AscendC::GlobalTensor<ElementB> gmBImagSigned;
        gmBImagSigned.SetGlobalBuffer((__gm__ ElementB*)params.ptrBImagSigned);
        AscendC::GlobalTensor<ElementCReal> gmCReal;
        gmCReal.SetGlobalBuffer((__gm__ ElementCReal*)params.ptrCReal);
        AscendC::GlobalTensor<ElementCImag> gmCImag;
        gmCImag.SetGlobalBuffer((__gm__ ElementCImag*)params.ptrCImag);

        // ---- Common TLA tensor setup (shared by both variants) ----
        // Build TLA concrete layouts from the tags stored in Params, then wrap
        // the GM buffers as tla::Tensors. Both 4-pass and fused paths tile
        // these via tla::GetTile and pass the resulting sub-tensors to their
        // respective BlockMmad, which takes TLA tensors directly.
        using LayoutTagA = typename Traits::LayoutA;
        using LayoutTagB = typename Traits::LayoutBReal;
        using LayoutTagC = typename Traits::LayoutC;

        constexpr uint32_t L1_TILE_M = tla::get<0>(L1TileShape{});
        constexpr uint32_t L1_TILE_N = tla::get<1>(L1TileShape{});
        constexpr uint32_t L1_TILE_K = tla::get<2>(L1TileShape{});

        auto tlaLayoutA = tla::MakeLayout<ElementA, LayoutTagA>(params.problemShape.m(), params.problemShape.k());
        auto tlaLayoutB = tla::MakeLayout<ElementB, LayoutTagB>(params.problemShape.k(), params.problemShape.n());
        auto tlaLayoutC = tla::MakeLayout<ElementCReal, LayoutTagC>(params.problemShape.m(), params.problemShape.n());

        auto tensorAReal = tla::MakeTensor(gmAReal, tlaLayoutA, Arch::PositionGM{});
        auto tensorAImag = tla::MakeTensor(gmAImag, tlaLayoutA, Arch::PositionGM{});
        auto tensorAImagSigned = tla::MakeTensor(gmAImagSigned, tlaLayoutA, Arch::PositionGM{});
        auto tensorBReal = tla::MakeTensor(gmBReal, tlaLayoutB, Arch::PositionGM{});
        auto tensorBImag = tla::MakeTensor(gmBImag, tlaLayoutB, Arch::PositionGM{});
        auto tensorBImagSigned = tla::MakeTensor(gmBImagSigned, tlaLayoutB, Arch::PositionGM{});
        auto tensorCReal = tla::MakeTensor(gmCReal, tlaLayoutC, Arch::PositionGM{});
        auto tensorCImag = tla::MakeTensor(gmCImag, tlaLayoutC, Arch::PositionGM{});

        BlockScheduler scheduler(params.problemShape, MakeCoord(L1_TILE_M, L1_TILE_N));
        uint32_t coreLoops = scheduler.GetCoreLoops();

        if constexpr (USE_FOUR_PASS) {
            using ElementAccumulator = typename Traits::ElementAccumulator;

            // Pass 1: C_real = A_real * B_real (overwrite)
            {
                BlockMmad blockMmad(resource);
                AscendC::SetAtomicNone();
                for (uint32_t loopIdx = AscendC::GetBlockIdx(); loopIdx < coreLoops;
                     loopIdx += AscendC::GetBlockNum()) {
                    GemmCoord blockCoord = scheduler.GetBlockCoord(loopIdx);
                    GemmCoord actualBlockShape = scheduler.GetActualBlockShape(blockCoord);
                    uint32_t mIndex = blockCoord.m() * L1_TILE_M;
                    uint32_t nIndex = blockCoord.n() * L1_TILE_N;
                    uint32_t kIndex = blockCoord.k() * L1_TILE_K;
                    auto tensorBlockA = tla::GetTile(
                        tensorAReal, tla::MakeCoord(mIndex, kIndex),
                        tla::MakeShape(actualBlockShape.m(), actualBlockShape.k()));
                    auto tensorBlockB = tla::GetTile(
                        tensorBReal, tla::MakeCoord(kIndex, nIndex),
                        tla::MakeShape(actualBlockShape.k(), actualBlockShape.n()));
                    auto tensorBlockC = tla::GetTile(
                        tensorCReal, tla::MakeCoord(mIndex, nIndex),
                        tla::MakeShape(actualBlockShape.m(), actualBlockShape.n()));
                    blockMmad(tensorBlockA, tensorBlockB, tensorBlockC, actualBlockShape);
                }
            }

            // Pass 2: C_real += signed imaginary cross term (atomic add)
            {
                BlockMmad blockMmad(resource);
                AscendC::SetAtomicAdd<ElementAccumulator>();
                for (uint32_t loopIdx = AscendC::GetBlockIdx(); loopIdx < coreLoops;
                     loopIdx += AscendC::GetBlockNum()) {
                    GemmCoord blockCoord = scheduler.GetBlockCoord(loopIdx);
                    GemmCoord actualBlockShape = scheduler.GetActualBlockShape(blockCoord);
                    uint32_t mIndex = blockCoord.m() * L1_TILE_M;
                    uint32_t nIndex = blockCoord.n() * L1_TILE_N;
                    uint32_t kIndex = blockCoord.k() * L1_TILE_K;
                    auto tensorBlockC = tla::GetTile(
                        tensorCReal, tla::MakeCoord(mIndex, nIndex),
                        tla::MakeShape(actualBlockShape.m(), actualBlockShape.n()));
                    if constexpr (NEGATE_A) {
                        auto tensorBlockA = tla::GetTile(
                            tensorAImagSigned, tla::MakeCoord(mIndex, kIndex),
                            tla::MakeShape(actualBlockShape.m(), actualBlockShape.k()));
                        auto tensorBlockB = tla::GetTile(
                            tensorBImag, tla::MakeCoord(kIndex, nIndex),
                            tla::MakeShape(actualBlockShape.k(), actualBlockShape.n()));
                        blockMmad(tensorBlockA, tensorBlockB, tensorBlockC, actualBlockShape);
                    } else {
                        auto tensorBlockA = tla::GetTile(
                            tensorAImag, tla::MakeCoord(mIndex, kIndex),
                            tla::MakeShape(actualBlockShape.m(), actualBlockShape.k()));
                        auto tensorBlockB = tla::GetTile(
                            tensorBImagSigned, tla::MakeCoord(kIndex, nIndex),
                            tla::MakeShape(actualBlockShape.k(), actualBlockShape.n()));
                        blockMmad(tensorBlockA, tensorBlockB, tensorBlockC, actualBlockShape);
                    }
                }
                AscendC::SetAtomicNone();
            }

            // Pass 3: C_imag = A_imag * B_real (overwrite)
            {
                BlockMmad blockMmad(resource);
                AscendC::SetAtomicNone();
                for (uint32_t loopIdx = AscendC::GetBlockIdx(); loopIdx < coreLoops;
                     loopIdx += AscendC::GetBlockNum()) {
                    GemmCoord blockCoord = scheduler.GetBlockCoord(loopIdx);
                    GemmCoord actualBlockShape = scheduler.GetActualBlockShape(blockCoord);
                    uint32_t mIndex = blockCoord.m() * L1_TILE_M;
                    uint32_t nIndex = blockCoord.n() * L1_TILE_N;
                    uint32_t kIndex = blockCoord.k() * L1_TILE_K;
                    auto tensorBlockA = tla::GetTile(
                        tensorAImag, tla::MakeCoord(mIndex, kIndex),
                        tla::MakeShape(actualBlockShape.m(), actualBlockShape.k()));
                    auto tensorBlockB = tla::GetTile(
                        tensorBReal, tla::MakeCoord(kIndex, nIndex),
                        tla::MakeShape(actualBlockShape.k(), actualBlockShape.n()));
                    auto tensorBlockC = tla::GetTile(
                        tensorCImag, tla::MakeCoord(mIndex, nIndex),
                        tla::MakeShape(actualBlockShape.m(), actualBlockShape.n()));
                    blockMmad(tensorBlockA, tensorBlockB, tensorBlockC, actualBlockShape);
                }
            }

            // Pass 4: C_imag += A_real * B_imag (atomic add)
            {
                BlockMmad blockMmad(resource);
                AscendC::SetAtomicAdd<ElementAccumulator>();
                for (uint32_t loopIdx = AscendC::GetBlockIdx(); loopIdx < coreLoops;
                     loopIdx += AscendC::GetBlockNum()) {
                    GemmCoord blockCoord = scheduler.GetBlockCoord(loopIdx);
                    GemmCoord actualBlockShape = scheduler.GetActualBlockShape(blockCoord);
                    uint32_t mIndex = blockCoord.m() * L1_TILE_M;
                    uint32_t nIndex = blockCoord.n() * L1_TILE_N;
                    uint32_t kIndex = blockCoord.k() * L1_TILE_K;
                    auto tensorBlockA = tla::GetTile(
                        tensorAReal, tla::MakeCoord(mIndex, kIndex),
                        tla::MakeShape(actualBlockShape.m(), actualBlockShape.k()));
                    auto tensorBlockB = tla::GetTile(
                        tensorBImag, tla::MakeCoord(kIndex, nIndex),
                        tla::MakeShape(actualBlockShape.k(), actualBlockShape.n()));
                    auto tensorBlockC = tla::GetTile(
                        tensorCImag, tla::MakeCoord(mIndex, nIndex),
                        tla::MakeShape(actualBlockShape.m(), actualBlockShape.n()));
                    blockMmad(tensorBlockA, tensorBlockB, tensorBlockC, actualBlockShape);
                }
                AscendC::SetAtomicNone();
            }
        } else {
            // Fused variant — single BlockMmadTla (MmadPlanarComplexFused) call per block,
            // alternating A/B operands with a shared L0C buffer. The block is
            // agnostic to NEGATE_A; it reads the Signed GM slots the kernel
            // aliases (workspace or original) for the C_real cross term.
            BlockMmad blockMmad(resource);

            for (uint32_t loopIdx = AscendC::GetBlockIdx(); loopIdx < coreLoops; loopIdx += AscendC::GetBlockNum()) {
                GemmCoord blockCoord = scheduler.GetBlockCoord(loopIdx);
                GemmCoord actualBlockShape = scheduler.GetActualBlockShape(blockCoord);

                uint32_t mIndex = blockCoord.m() * L1_TILE_M;
                uint32_t nIndex = blockCoord.n() * L1_TILE_N;
                uint32_t kIndex = blockCoord.k() * L1_TILE_K;
                uint32_t mSize = actualBlockShape.m();
                uint32_t nSize = actualBlockShape.n();
                uint32_t kSize = actualBlockShape.k();

                auto tensorBlockAReal =
                    tla::GetTile(tensorAReal, tla::MakeCoord(mIndex, kIndex), tla::MakeShape(mSize, kSize));
                auto tensorBlockAImag =
                    tla::GetTile(tensorAImag, tla::MakeCoord(mIndex, kIndex), tla::MakeShape(mSize, kSize));
                auto tensorBlockAImagSigned =
                    tla::GetTile(tensorAImagSigned, tla::MakeCoord(mIndex, kIndex), tla::MakeShape(mSize, kSize));
                auto tensorBlockBReal =
                    tla::GetTile(tensorBReal, tla::MakeCoord(kIndex, nIndex), tla::MakeShape(kSize, nSize));
                auto tensorBlockBImag =
                    tla::GetTile(tensorBImag, tla::MakeCoord(kIndex, nIndex), tla::MakeShape(kSize, nSize));
                auto tensorBlockBImagSigned =
                    tla::GetTile(tensorBImagSigned, tla::MakeCoord(kIndex, nIndex), tla::MakeShape(kSize, nSize));
                auto tensorBlockCReal =
                    tla::GetTile(tensorCReal, tla::MakeCoord(mIndex, nIndex), tla::MakeShape(mSize, nSize));
                auto tensorBlockCImag =
                    tla::GetTile(tensorCImag, tla::MakeCoord(mIndex, nIndex), tla::MakeShape(mSize, nSize));

                blockMmad(
                    tensorBlockAReal, tensorBlockAImag, tensorBlockAImagSigned, tensorBlockBReal, tensorBlockBImag,
                    tensorBlockBImagSigned, tensorBlockCReal, tensorBlockCImag, actualBlockShape);
            }
        }

        AscendC::PipeBarrier<PIPE_ALL>();
    }

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIV>(Params const& params)
    {
        using NegateElement = std::conditional_t<NEGATE_A, ElementA, ElementB>;

        AscendC::GlobalTensor<NegateElement> gmSrc;
        AscendC::GlobalTensor<NegateElement> gmDst;
        uint32_t totalElements = 0;
        if constexpr (NEGATE_A) {
            gmSrc.SetGlobalBuffer((__gm__ NegateElement*)params.ptrAImag);
            gmDst.SetGlobalBuffer((__gm__ NegateElement*)params.ptrAImagSigned);
            totalElements = params.layoutA.Capacity();
        } else {
            gmSrc.SetGlobalBuffer((__gm__ NegateElement*)params.ptrBImag);
            gmDst.SetGlobalBuffer((__gm__ NegateElement*)params.ptrBImagSigned);
            totalElements = params.layoutBImag.Capacity();
        }
        gmSrc.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        gmDst.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);

        Arch::Resource<ArchTag> resource;
        NegateMatrixAiv<ArchTag, NegateElement> negate(resource);
        negate(gmDst, gmSrc, totalElements);

        Catlass::Arch::CrossCoreBarrier<0x0, PIPE_MTE3>();
        Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(flagAivFinishNegate);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

private:
    static constexpr Arch::FlagID FLAG_AIV_FINISH_NEGATE = 0;
    Arch::CrossCoreFlag flagAivFinishNegate{FLAG_AIV_FINISH_NEGATE};
};

} // namespace Catlass::Gemm::Kernel

#endif // CATLASS_GEMM_KERNEL_PLANAR_COMPLEX_GEMM_TLA_HPP
