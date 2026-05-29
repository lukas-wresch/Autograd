#pragma once
#include <stdexcept>
#include "tensor.h"



class Kernels
{
public:
	static void Multiply_Forward( Tensor& out, const Tensor& left, const Tensor& right);
	static void Multiply_Backward(Tensor& out, const Tensor& left, const Tensor& right);

    static void MatMul_Forward( Tensor& out, const Tensor& left, const Tensor& right);
    static void MatMul_Backward(Tensor& out, const Tensor& left, const Tensor& right);
    static inline void MatMul_Backward_A(Tensor& gradA, const Tensor& dOut, const Tensor& B);
    static inline void MatMul_Backward_B(Tensor& gradB, const Tensor& A, const Tensor& dOut);

    static void Tanh_Backward(Tensor& grad_in, const Tensor& out, const Tensor& outer_grad);
};



inline static void DecodeIndex(size_t linear, const std::vector<size_t>& shape, std::vector<size_t>& outIdx)
{
    for (int d = (int)shape.size() - 1; d >= 0; d--)
    {
        outIdx[d] = linear % shape[d];
        linear /= shape[d];
    }
}



inline static size_t ComputeOffset(const std::vector<size_t>& idx, const std::vector<size_t>& strides)
{
    size_t off = 0;

    for (size_t i = 0; i < idx.size(); i++)
        off += idx[i] * strides[i];

    return off;
}



inline void Kernels::Multiply_Forward(Tensor& out, const Tensor& a, const Tensor& b)
{
    if (!IsBroadcastCompatible(a.Shape(), b.Shape()))
        throw std::runtime_error("Kernels::Multiply_Forward() Broadcast mismatch");

    auto outShape = BroadcastShape(a.Shape(), b.Shape());

    if (out.Shape() != outShape)
        throw std::runtime_error("Kernels::Multiply_Forward() output shape mismatch");

    const auto aStr = a.BroadcastStrides(outShape);
    const auto bStr = b.BroadcastStrides(outShape);

    const auto& oStr = out.Strides();

    std::vector<size_t> idx(outShape.size());

    const float* A = a.Data();
    const float* B = b.Data();
    float* O = out.Data();

    const size_t total = out.Size();

    for (size_t linear = 0; linear < total; linear++)
    {
        DecodeIndex(linear, outShape, idx);

        size_t ao = ComputeOffset(idx, aStr);
        size_t bo = ComputeOffset(idx, bStr);
        size_t oo = ComputeOffset(idx, oStr);

        O[oo] = A[ao] * B[bo];
    }
}



inline void Kernels::MatMul_Forward(Tensor& out, const Tensor& A, const Tensor& B)
{
    const auto& ashape = A.Shape();
    const auto& bshape = B.Shape();
    const auto& oshape = out.Shape();

    const size_t ndim = ashape.size();

    if (ndim < 2 || bshape.size() != ndim)
        throw std::runtime_error(
            "Kernels::MatMul_Forward() invalid dimensions");

    size_t M = ashape[ndim - 2];
    size_t K = ashape[ndim - 1];

    size_t K2 = bshape[ndim - 2];
    size_t N = bshape[ndim - 1];

    if (K != K2)
        throw std::runtime_error(
            "Kernels::MatMul_Forward() K mismatch");

    std::vector<size_t> expected = ashape;
    expected[ndim - 1] = N;

    if (oshape != expected)
        throw std::runtime_error(
            "Kernels::MatMul_Forward() output shape mismatch");

    const auto& aStr = A.Strides();
    const auto& bStr = B.Strides();
    const auto& oStr = out.Strides();

    const float* aData = A.Data();
    const float* bData = B.Data();
    float* oData = out.Data();

    std::vector<size_t> idx(ndim);

    size_t total = out.Size();

    for (size_t linear = 0; linear < total; linear++)
    {
        DecodeIndex(linear, oshape, idx);

        size_t outOffset = ComputeOffset(idx, oStr);

        size_t i = idx[ndim - 2];
        size_t j = idx[ndim - 1];

        float sum = 0.0f;

        for (size_t k = 0; k < K; k++)
        {
            auto aIdx = idx;
            auto bIdx = idx;

            aIdx[ndim - 1] = k;

            bIdx[ndim - 2] = k;
            bIdx[ndim - 1] = j;

            size_t ao = ComputeOffset(aIdx, aStr);
            size_t bo = ComputeOffset(bIdx, bStr);

            sum += aData[ao] * bData[bo];
        }

        oData[outOffset] = sum;
    }
}



inline void Kernels::Tanh_Backward(Tensor& grad_in, const Tensor& out, const Tensor& outer_grad)
{
    if (!IsBroadcastCompatible(out.Shape(), outer_grad.Shape()))
        throw std::runtime_error("Kernels::Tanh_Backward() broadcast mismatch");

    auto shape = BroadcastShape(out.Shape(), outer_grad.Shape());

    if (grad_in.Shape() != shape)
        throw std::runtime_error("Kernels::Tanh_Backward() grad shape mismatch");

    auto outStr = out.BroadcastStrides(shape);
    auto grdStr = outer_grad.BroadcastStrides(shape);

    std::vector<size_t> idx(shape.size());

    size_t total = grad_in.Size();

    for (size_t i = 0; i < total; i++)
    {
        size_t tmp = i;

        // linear -> ND index
        for (int d = (int)shape.size() - 1; d >= 0; d--)
        {
            idx[d] = tmp % shape[d];
            tmp /= shape[d];
        }

        size_t oo = out.Offset();
        size_t go = outer_grad.Offset();

        for (size_t d = 0; d < shape.size(); d++)
        {
            oo += idx[d] * outStr[d];
            go += idx[d] * grdStr[d];
        }

        float t = out.Data()[oo];

        grad_in.Data()[i] += (1.0f - t * t) * outer_grad.Data()[go];
    }
}



// dA += dOut @ B^T
inline void Kernels::MatMul_Backward_A(
    Tensor& gradA,
    const Tensor& dOut,
    const Tensor& B)
{
    const auto& ashape = gradA.Shape();
    const auto& oshape = dOut.Shape();
    const auto& bshape = B.Shape();

    const size_t ndim = ashape.size();

    size_t M = ashape[ndim - 2];
    size_t K = ashape[ndim - 1];
    size_t N = bshape[ndim - 1];

    const auto& aStr = gradA.Strides();
    const auto& oStr = dOut.Strides();
    const auto& bStr = B.Strides();

    float* aGrad = gradA.Data();

    const float* outGrad = dOut.Data();
    const float* bData = B.Data();

    std::vector<size_t> idx(ndim);

    size_t total = gradA.Size();

    for (size_t linear = 0; linear < total; linear++)
    {
        DecodeIndex(linear, ashape, idx);

        size_t i = idx[ndim - 2];
        size_t k = idx[ndim - 1];

        float sum = 0.0f;

        for (size_t j = 0; j < N; j++)
        {
            auto oIdx = idx;
            auto bIdx = idx;

            oIdx[ndim - 2] = i;
            oIdx[ndim - 1] = j;

            bIdx[ndim - 2] = k;
            bIdx[ndim - 1] = j;

            size_t oo = ComputeOffset(oIdx, oStr);
            size_t bo = ComputeOffset(bIdx, bStr);

            sum += outGrad[oo] * bData[bo];
        }

        size_t ao = ComputeOffset(idx, aStr);

        aGrad[ao] += sum;
    }
}



inline void Kernels::MatMul_Backward_B(
    Tensor& gradB,
    const Tensor& A,
    const Tensor& dOut)
{
    const auto& bshape = gradB.Shape();
    const auto& ashape = A.Shape();
    const auto& oshape = dOut.Shape();

    const size_t ndim = bshape.size();

    size_t K = bshape[ndim - 2];
    size_t N = bshape[ndim - 1];
    size_t M = ashape[ndim - 2];

    const auto& bStr = gradB.Strides();
    const auto& aStr = A.Strides();
    const auto& oStr = dOut.Strides();

    float* bGrad = gradB.Data();

    const float* aData = A.Data();
    const float* outGrad = dOut.Data();

    std::vector<size_t> idx(ndim);

    size_t total = gradB.Size();

    for (size_t linear = 0; linear < total; linear++)
    {
        DecodeIndex(linear, bshape, idx);

        size_t k = idx[ndim - 2];
        size_t j = idx[ndim - 1];

        float sum = 0.0f;

        for (size_t i = 0; i < M; i++)
        {
            auto aIdx = idx;
            auto oIdx = idx;

            aIdx[ndim - 2] = i;
            aIdx[ndim - 1] = k;

            oIdx[ndim - 2] = i;
            oIdx[ndim - 1] = j;

            size_t ao = ComputeOffset(aIdx, aStr);
            size_t oo = ComputeOffset(oIdx, oStr);

            sum += aData[ao] * outGrad[oo];
        }

        size_t bo = ComputeOffset(idx, bStr);

        bGrad[bo] += sum;
    }
}