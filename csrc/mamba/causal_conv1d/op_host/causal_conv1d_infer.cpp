// Shape/type inference does not read tensor values or depend on Torch.
#include "register/op_impl_registry.h"
#include <string>

namespace ops {
static ge::graphStatus InferShapeCausalConv1d(gert::InferShapeContext *context)
{
    const auto *x = context->GetInputShape(0);
    const auto *state = context->GetInputShape(2);
    auto *y = context->GetOutputShape(0);
    auto *stateRef = context->GetOutputShape(1);
    if (!x || !state || !y || !stateRef) return ge::GRAPH_FAILED;
    const auto *attrs = context->GetAttrs();
    const auto *mode = attrs ? attrs->GetAttrPointer<int64_t>(2) : nullptr;
    const char *layout = attrs ? attrs->GetStr(3) : nullptr;
    const auto *heads = attrs ? attrs->GetAttrPointer<int64_t>(4) : nullptr;
    const std::string outputLayout = layout ? layout : "TOKEN_FIRST";
    const int64_t headNum = heads ? *heads : 0;
    if (outputLayout == "HEAD_FIRST") {
        if ((mode && *mode != 0) || headNum <= 0 || (x->GetDimNum() != 2 && x->GetDimNum() != 3))
            return ge::GRAPH_FAILED;
        const int64_t channels = x->GetDim(x->GetDimNum() - 1);
        if (channels >= 0 && (channels == 0 || channels % headNum != 0 || (channels / headNum) % 16 != 0))
            return ge::GRAPH_FAILED;
        const int64_t headDim = channels < 0 ? -1 : channels / headNum;
        if (x->GetDimNum() == 2) {
            y->SetDimNum(3);
            y->SetDim(0, headNum);
            y->SetDim(1, x->GetDim(0));
            y->SetDim(2, headDim);
        } else {
            y->SetDimNum(4);
            y->SetDim(0, x->GetDim(0));
            y->SetDim(1, headNum);
            y->SetDim(2, x->GetDim(1));
            y->SetDim(3, headDim);
        }
    } else {
        if (outputLayout != "TOKEN_FIRST" || headNum != 0) return ge::GRAPH_FAILED;
        *y = *x;
    }
    *stateRef = *state;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeCausalConv1d(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    context->SetOutputDataType(1, context->GetInputDataType(2));
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(SglCausalConv1d).InferShape(InferShapeCausalConv1d).InferDataType(InferDataTypeCausalConv1d);
}  // namespace ops
