/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_EPILOGUE_TILE_TILE_ELEMWISE_GELU_HPP
#define CATLASS_EPILOGUE_TILE_TILE_ELEMWISE_GELU_HPP

#include "catlass/catlass.hpp"
#include "catlass/matrix_coord.hpp"

#if (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510)
using namespace AscendC::Reg;
#endif

namespace Catlass::Epilogue::Tile {
template <
    // / Tag indicating architecture
    class ArchTag_,
    // / Compute data type
    class ComputeType_,
    // / Length of the compute buffer
    uint32_t COMPUTE_LENGTH_>
struct TileElemWiseGelu {
    using ArchTag = ArchTag_;
    using ElementCompute = typename ComputeType_::Element;

    static constexpr uint32_t COMPUTE_LENGTH = COMPUTE_LENGTH_;
    static constexpr float NEG_SQRT_EIGHT_OVER_PI = -1.595769121 * 0.044715;
    static constexpr float TANH_APPROX_FACTOR = 1 / 0.044715;

    CATLASS_DEVICE
    TileElemWiseGelu()
    {}

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<ElementCompute> const& dstLocal, AscendC::LocalTensor<ElementCompute> const& srcLocal)
    {
        using namespace AscendC;

        // current realization: x / (1 + e^(-1.5957691*0.044715(x/0.044715 + x^3)))
        Mul(dstLocal, srcLocal, srcLocal, COMPUTE_LENGTH);            // d: x^2 , s:x
        Mul(dstLocal, dstLocal, srcLocal, COMPUTE_LENGTH);            // d: x^3 ,.s:x
        Axpy(dstLocal, srcLocal, TANH_APPROX_FACTOR, COMPUTE_LENGTH); // d: x / 0.044715 + x^3 , s: x
        // d: -1.5957691*0.044715(x/0.044715 + x^3), s: x
        Muls(dstLocal, dstLocal, NEG_SQRT_EIGHT_OVER_PI, COMPUTE_LENGTH);
        Exp(dstLocal, dstLocal, COMPUTE_LENGTH); // d: e^(-1.5957691*0.044715(x/0.044715 + x^3))
        // d: (1 + e^(-1.5957691*0.044715(x/0.044715 + x^3))
        Adds(dstLocal, dstLocal, (ElementCompute)1, COMPUTE_LENGTH);
        Div(dstLocal, srcLocal, dstLocal, COMPUTE_LENGTH);
    }
};

#if (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510)
template <
    // / Tag indicating architecture
    class ArchTag_,
    // / Compute data type
    class ElementDst_, class ElementSrc_>
struct TileElemWiseGeluRegBase {
    using ArchTag = ArchTag_;
    using ElementDst = ElementDst_;
    using ElementSrc = ElementSrc_;

    static_assert(std::is_same_v<ElementSrc, float32_t>, "ElementSrc must be float32_t");
    static_assert(std::is_same_v<ElementDst, half>, "ElementDst must be half");

    static constexpr float NEG_SQRT_EIGHT_OVER_PI = -1.595769121 * 0.044715;
    static constexpr float TANH_APPROX_FACTOR = 1 / 0.044715;

    CATLASS_DEVICE
    TileElemWiseGeluRegBase()
    {}

    __simd_vf__ static void GeluVf(
        __ubuf__ ElementDst* dstUb, __ubuf__ ElementSrc* srcUb, uint32_t actualRowNum, uint32_t actualColumnNum,
        uint32_t dstRowStride, uint32_t srcRowStride)
    {
        static constexpr CastTrait castTraitB32ToB16 = {
            RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};

        static constexpr uint16_t vlSize = static_cast<uint16_t>(AscendC::GetVecLen() / sizeof(ElementSrc));
        uint16_t loops = AscendC::CeilDivision(actualColumnNum, vlSize);

        __ubuf__ ElementSrc* srcRowStart{nullptr};
        __ubuf__ ElementDst* dstRowStart{nullptr};

        MaskReg pregFullB32 = CreateMask<float, MaskPattern::ALL>();
        RegTensor<ElementSrc> srcVreg;
        RegTensor<float> computeVreg;
        RegTensor<ElementDst> dstVreg;

        for (uint16_t rowIdx = 0; rowIdx < actualRowNum; rowIdx++) {
            srcRowStart = srcUb + rowIdx * srcRowStride;
            dstRowStart = dstUb + rowIdx * dstRowStride;

            for (uint16_t loopIdx = 0; loopIdx < loops; loopIdx++) {
                uint32_t offset = loopIdx * vlSize;

                LoadAlign<ElementSrc>(srcVreg, srcRowStart + offset);
                Mul(computeVreg, srcVreg, srcVreg, pregFullB32);             // d: x^2 , s:x
                Mul(computeVreg, computeVreg, srcVreg, pregFullB32);         // d: x^3 , s:x
                Axpy(computeVreg, srcVreg, TANH_APPROX_FACTOR, pregFullB32); // d: x / 0.044715 + x^3 , s: x
                Muls(
                    computeVreg, computeVreg, NEG_SQRT_EIGHT_OVER_PI,
                    pregFullB32);                           // d: -1.5957691*0.044715(x/0.044715 + x^3), s: x
                Exp(computeVreg, computeVreg, pregFullB32); // d: e^(-1.5957691*0.044715(x/0.044715 + x^3))
                Adds(
                    computeVreg, computeVreg, (float)1,
                    pregFullB32); // d: (1 + e^(-1.5957691*0.044715(x/0.044715 + x^3)))
                Div(computeVreg, srcVreg, computeVreg,
                    pregFullB32); // d: x / (1 + e^(-1.5957691*0.044715(x/0.044715 + x^3)))

                Cast<ElementDst, float, castTraitB32ToB16>(dstVreg, computeVreg, pregFullB32);
                StoreAlign<ElementDst, StoreDist::DIST_PACK_B32>(dstRowStart + offset, dstVreg, pregFullB32);
            }
        }
    }

    template <class LayoutTagDst, class LayoutTagSrc>
    CATLASS_DEVICE void operator()(
        AscendC::LocalTensor<ElementDst> const& ubDst, AscendC::LocalTensor<ElementSrc> const& ubSrc,
        LayoutTagDst const& layoutTagDst, LayoutTagSrc const& layoutTagSrc, MatrixCoord const& actualBlockShape)
    {
        __ubuf__ ElementDst* ubDstPtr = (__ubuf__ ElementDst*)ubDst.GetPhyAddr();
        __ubuf__ ElementSrc* ubSrcPtr = (__ubuf__ ElementSrc*)ubSrc.GetPhyAddr();
        uint32_t actualRowNum = actualBlockShape.row();
        uint32_t actualColumnNum = actualBlockShape.column();

        uint32_t dstRowStride = layoutTagDst.stride(0);
        uint32_t srcRowStride = layoutTagSrc.stride(0);

        GeluVf(ubDstPtr, ubSrcPtr, actualRowNum, actualColumnNum, dstRowStride, srcRowStride);
    }
};
#endif

} // namespace Catlass::Epilogue::Tile

#endif
