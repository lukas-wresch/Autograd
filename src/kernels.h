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
};



inline bool IsBroadcastCompatible(const std::vector<size_t>& a, const std::vector<size_t>& b)
{
    size_t aDim = a.size();
    size_t bDim = b.size();
    size_t maxDim = std::max(aDim, bDim);

    for (size_t i = 0; i < maxDim; i++)
    {
        size_t ad = (i < aDim)
            ? a[aDim - 1 - i]
            : 1;

        size_t bd = (i < bDim)
            ? b[bDim - 1 - i]
            : 1;

        if (ad != bd && ad != 1 && bd != 1)
            return false;
    }

    return true;
}



inline void Kernels::Multiply_Forward(
    Tensor& out,
    const Tensor& a,
    const Tensor& b)
{
    if (!IsBroadcastCompatible(a.Shape(), b.Shape()))
        throw std::runtime_error("Broadcast mismatch");

    const auto& shape = out.Shape();
    size_t total = out.Size();

    auto aStr = a.BroadcastStrides(shape);
    auto bStr = b.BroadcastStrides(shape);

    std::vector<size_t> idx(shape.size());

    for (size_t i = 0; i < total; i++)
    {
        size_t tmp = i;

        for (int d = (int)shape.size() - 1; d >= 0; d--)
        {
            idx[d] = tmp % shape[d];
            tmp /= shape[d];
        }

        size_t ao = a.Offset();
        size_t bo = b.Offset();

        for (size_t d = 0; d < shape.size(); d++)
        {
            ao += idx[d] * aStr[d];
            bo += idx[d] * bStr[d];
        }

        out.Data()[i] = a.Data()[ao] * b.Data()[bo];
    }
}



inline void Kernels::MatMul_Forward(
    Tensor& out,
    const Tensor& a,
    const Tensor& b)
{
    const auto& ashape = a.Shape();
    const auto& bshape = b.Shape();
    const auto& oshape = out.Shape();

    const size_t ndim = oshape.size();

    if (ndim < 2)
        throw std::runtime_error("MatMul requires at least 2D tensors");

    size_t M = oshape[ndim - 2];
    size_t N = oshape[ndim - 1];
    size_t K = ashape[ndim - 1];

    if (bshape[ndim - 2] != K)
        throw std::runtime_error("MatMul shape mismatch");

    const float* A = a.Data();
    const float* B = b.Data();
    float* C = out.Data();

    // precompute strides
    auto aStr = a.Strides();
    auto bStr = b.Strides();
    auto oStr = out.Strides();

    size_t total = out.Size();

    for (size_t idx = 0; idx < total; idx++)
    {
        // decode flat index → ND index
        size_t tmp = idx;
        std::vector<size_t> index(ndim);

        for (int d = (int)ndim - 1; d >= 0; d--)
        {
            index[d] = tmp % oshape[d];
            tmp /= oshape[d];
        }

        // split indices
        size_t aOffset = a.Offset();
        size_t bOffset = b.Offset();

        // batch + matrix dims
        for (size_t d = 0; d < ndim - 2; d++)
        {
            size_t ai = (d < ashape.size()) ? index[d] : 0;
            size_t bi = (d < bshape.size()) ? index[d] : 0;

            aOffset += ai * aStr[d];
            bOffset += bi * bStr[d];
        }

        size_t i = index[ndim - 2];
        size_t j = index[ndim - 1];

        float sum = 0.0f;

        for (size_t k = 0; k < K; k++)
        {
            size_t aIdx = aOffset + i * aStr[ndim - 2] + k * aStr[ndim - 1];
            size_t bIdx = bOffset + k * bStr[ndim - 2] + j * bStr[ndim - 1];

            sum += A[aIdx] * B[bIdx];
        }

        C[idx] = sum;
    }
}