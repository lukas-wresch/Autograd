#pragma once
#include <stdexcept>
#include "tensor4d.h"
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

    static void SGD_Update(Tensor& param, const Tensor& grad, Tensor& velocity, float lr, float momentum);

    static Tensor4D Conv2D_Forward(const Tensor4D& Input, const Tensor4D& Kernel, int Stride = 1, int Padding = 0);
    static void Conv2D_Backward(Tensor4D& dInput, Tensor4D& dKernel, const Tensor4D& Input, const Tensor4D& Kernel, const Tensor4D& dOut, int Stride, int Padding);
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
        throw std::runtime_error("MatMul invalid dimensions");

    size_t M = ashape[ndim - 2];
    size_t K = ashape[ndim - 1];
    size_t K2 = bshape[ndim - 2];
    size_t N = bshape[ndim - 1];

    if (K != K2)
        throw std::runtime_error("K mismatch");

    std::vector<size_t> expected = ashape;
    expected[ndim - 1] = N;

    if (oshape != expected)
        throw std::runtime_error("Shape mismatch");

    const float* aData = A.Data();
    const float* bData = B.Data();
    float* oData = out.Data();

    const auto& aStr = A.Strides();
    const auto& bStr = B.Strides();
    const auto& oStr = out.Strides();

    auto isContiguous2D = [&]()
    {
        if (ndim != 2) return false;

        // classic row-major check
        return (aStr[1] == 1 && aStr[0] == K) &&
               (bStr[1] == 1 && bStr[0] == N) &&
               (oStr[1] == 1 && oStr[0] == N);
    };

    if (isContiguous2D())
    {
        // =========================
        // FAST PATH (2D contiguous)
        // =========================

        for (size_t i = 0; i < M; ++i)
        {
            const float* aRow = aData + i * K;
            float* oRow = oData + i * N;

            std::fill(oRow, oRow + N, 0.0f);

            for (size_t k = 0; k < K; ++k)
            {
                float aVal = aRow[k];
                const float* bRow = bData + k * N;

                for (size_t j = 0; j < N; ++j)
                {
                    oRow[j] += aVal * bRow[j];
                }
            }
        }

        return;
    }

    // =========================
    // GENERIC PATH (N-D tensors)
    // =========================

    std::vector<size_t> idx(ndim);

    size_t total = out.Size();

    for (size_t linear = 0; linear < total; ++linear)
    {
        DecodeIndex(linear, oshape, idx);

        size_t outOffset = ComputeOffset(idx, oStr);

        const size_t i = idx[ndim - 2];
        const size_t j = idx[ndim - 1];

        float sum = 0.0f;

        for (size_t k = 0; k < K; ++k)
        {
            idx[ndim - 1] = k;
            size_t aOffset = ComputeOffset(idx, aStr);

            idx[ndim - 2] = k;
            idx[ndim - 1] = j;
            size_t bOffset = ComputeOffset(idx, bStr);

            sum += aData[aOffset] * bData[bOffset];
        }

        oData[outOffset] = sum;
    }
}



/*inline void Kernels::MatMul_Forward(Tensor& out, const Tensor& A, const Tensor& B)
{
    const auto& ashape = A.Shape();
    const auto& bshape = B.Shape();
    const auto& oshape = out.Shape();

    const size_t ndim = ashape.size();

    if (ndim < 2 || bshape.size() != ndim)
        throw std::runtime_error("Kernels::MatMul_Forward() invalid dimensions");

    size_t M = ashape[ndim - 2];
    size_t K = ashape[ndim - 1];

    size_t K2 = bshape[ndim - 2];
    size_t N = bshape[ndim - 1];

    if (K != K2)
        throw std::runtime_error("Kernels::MatMul_Forward() K mismatch");

    std::vector<size_t> expected = ashape;
    expected[ndim - 1] = N;

    if (oshape != expected)
        throw std::runtime_error("Kernels::MatMul_Forward() output shape mismatch");

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
}*/



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
inline void Kernels::MatMul_Backward_A(Tensor& gradA, const Tensor& dOut, const Tensor& B)
{
    const auto& sA = gradA.Shape();
    const size_t ndim = sA.size();

    const size_t M = sA[ndim - 2];
    const size_t K = sA[ndim - 1];
    const size_t N = B.Shape()[ndim - 1];

    const float* dOutPtr = dOut.Data();
    const float* bPtr = B.Data();
    float* gPtr = gradA.Data();

    const auto& aStr = gradA.Strides();
    const auto& oStr = dOut.Strides();
    const auto& bStr = B.Strides();

    // iterate full tensor
    for (size_t linear = 0; linear < gradA.Size(); ++linear)
    {
        size_t tmp = linear;

        // decode only last 2 dims logically
        size_t base = linear / (M * K);
        size_t ik = linear % (M * K);

        size_t i = ik / K;
        size_t k = ik % K;

        float sum = 0.0f;

        for (size_t j = 0; j < N; ++j)
        {
            size_t o = base * oStr[0]
                + i * oStr[ndim - 2]
                + j * oStr[ndim - 1];

            size_t b = base * bStr[0]
                + k * bStr[ndim - 2]
                + j * bStr[ndim - 1];

            sum += dOutPtr[o] * bPtr[b];
        }

        gPtr[linear] += sum;
    }
}



//gradB += A.Transpose() @ dOut
inline void Kernels::MatMul_Backward_B(
    Tensor& gradB,
    const Tensor& A,
    const Tensor& dOut)
{
    const auto& bshape = gradB.Shape();
    const auto& ashape = A.Shape();
    const auto& oshape = dOut.Shape();

    const size_t ndim = bshape.size();

    const size_t K = bshape[ndim - 2];
    const size_t N = bshape[ndim - 1];
    const size_t M = ashape[ndim - 2];

    const float* aData = A.Data();
    const float* oData = dOut.Data();
    float* gData = gradB.Data();

    const auto& aStr = A.Strides();
    const auto& oStr = dOut.Strides();
    const auto& gStr = gradB.Strides();

    const size_t outerSize = gradB.Size() / (K * N);

    for (size_t base = 0; base < outerSize; ++base)
    {
        const size_t baseA = base * aStr[0];
        const size_t baseO = base * oStr[0];
        const size_t baseG = base * gStr[0];

        for (size_t k = 0; k < K; ++k)
        {
            const size_t gRowBase = baseG + k * gStr[ndim - 2];
            const size_t aColBase = baseA + k * aStr[ndim - 1];

            for (size_t j = 0; j < N; ++j)
            {
                const size_t gIdx = gRowBase + j * gStr[ndim - 1];
                const size_t oColBase = baseO + j * oStr[ndim - 1];

                float sum = 0.0f;

                for (size_t i = 0; i < M; ++i)
                {
                    const size_t aIdx = baseA
                        + i * aStr[ndim - 2]
                        + k * aStr[ndim - 1];

                    const size_t oIdx = baseO
                        + i * oStr[ndim - 2]
                        + j * oStr[ndim - 1];

                    sum += aData[aIdx] * oData[oIdx];
                }

                gData[gIdx] += sum;
            }
        }
    }
}



inline void Kernels::SGD_Update(Tensor& param, const Tensor& grad, Tensor& velocity, float lr, float momentum)
{
    float* p = param.Data();
    const float* g = grad.Data();
    float* v = velocity.Data();

    const size_t n = param.Size();

    for (size_t i = 0; i < n; ++i)
    {
        v[i] = momentum * v[i] - lr * g[i];
        p[i] += v[i];
    }
}



inline Tensor4D Kernels::Conv2D_Forward(const Tensor4D& Input, const Tensor4D& Kernel, int Stride, int Padding)
{
    // Output shape berechnen
    size_t N    = Input.GetShape()[0];
    size_t C_in = Input.GetShape()[1];
    size_t H    = Input.GetShape()[2];
    size_t W    = Input.GetShape()[3];

    size_t out_channels = Kernel.GetShape()[0];
    size_t in_channels  = Kernel.GetShape()[1];
    size_t kernel_size  = Kernel.GetShape()[2];

    size_t H_out = (H + 2 * Padding - kernel_size) / Stride + 1;
    size_t W_out = (W + 2 * Padding - kernel_size) / Stride + 1;

    auto output = Tensor4D({ N, out_channels, H_out, W_out });

    for (size_t n = 0; n < N; n++)
    {
        for (size_t oc = 0; oc < out_channels; oc++)
        {
            for (size_t oh = 0; oh < H_out; oh++)
            {
                for (size_t ow = 0; ow < W_out; ow++)
                {
                    float sum = 0.0f;

                    for (size_t ic = 0; ic < in_channels; ic++)
                    {
                        for (size_t kh = 0; kh < kernel_size; kh++)
                        {
                            for (size_t kw = 0; kw < kernel_size; kw++)
                            {
                                int ih = (int)(oh * Stride + kh - Padding);
                                int iw = (int)(ow * Stride + kw - Padding);

                                if (ih >= 0 && ih < (int)H && iw >= 0 && iw < (int)W)
                                    sum += Input.At(n, ic, ih, iw) * Kernel.At(oc, ic, kh, kw);
                            }
                        }
                    }

                    output.At(n, oc, oh, ow) = sum;
                }
            }
        }
    }

    return output;
}



inline void Kernels::Conv2D_Backward(Tensor4D& dInput, Tensor4D& dKernel, const Tensor4D& Input, const Tensor4D& Kernel, const Tensor4D& dOut, int Stride, int Padding)
{
    if (Input.GetShape()[0] != dOut.GetShape()[0])
        throw std::runtime_error("Conv2D_Backward(): Batch size mismatch");
    if (Input.GetShape()[1] != Kernel.GetShape()[1])
        throw std::runtime_error("Conv2D_Backward(): Input channels != Kernel in_channels");
    if (Kernel.GetShape()[0] != dOut.GetShape()[1])
        throw std::runtime_error("Conv2D_Backward(): Kernel out_channels != dOut channels");
    if (Input.GetShape()[2] == 0 || Input.GetShape()[3] == 0)
        throw std::runtime_error("Conv2D_Backward(): Invalid input spatial size");
    if (Kernel.GetShape()[2] == 0 || Kernel.GetShape()[3] == 0)
        throw std::runtime_error("Conv2D_Backward(): Invalid kernel size");

    dInput.SetZero();
    dKernel.SetZero();

    const size_t N = Input.GetShape()[0];
    const size_t Cin = Input.GetShape()[1];
    const size_t H = Input.GetShape()[2];
    const size_t W = Input.GetShape()[3];

    const size_t Cout = Kernel.GetShape()[0];
    const size_t K = Kernel.GetShape()[2];

    const size_t Hout = dOut.GetShape()[2];
    const size_t Wout = dOut.GetShape()[3];

    size_t expectedH = (H + 2 * Padding - K) / Stride + 1;
    size_t expectedW = (W + 2 * Padding - K) / Stride + 1;

    if (dOut.GetShape()[2] != expectedH || dOut.GetShape()[3] != expectedW)
        throw std::runtime_error("Conv2D_Backward(): dOut spatial shape mismatch");

    //----------------------------------------------------
    // dKernel
    //----------------------------------------------------

    for (size_t n = 0; n < N; n++)
    {
        for (size_t oc = 0; oc < Cout; oc++)
        {
            for (size_t ic = 0; ic < Cin; ic++)
            {
                for (size_t kh = 0; kh < K; kh++)
                {
                    for (size_t kw = 0; kw < K; kw++)
                    {
                        float grad = 0.0f;

                        for (size_t oh = 0; oh < Hout; oh++)
                        {
                            for (size_t ow = 0; ow < Wout; ow++)
                            {
                                int ih = (int)(oh * Stride + kh - Padding);
                                int iw = (int)(ow * Stride + kw - Padding);

                                if (ih >= 0 && ih < (int)H && iw >= 0 && iw < (int)W)
                                {
                                    grad += Input.At(n, ic, ih, iw) * dOut.At(n, oc, oh, ow);
                                }
                            }
                        }

                        dKernel.At(oc, ic, kh, kw) += grad;
                    }
                }
            }
        }
    }

    //----------------------------------------------------
    // dInput
    //----------------------------------------------------

    for (size_t n = 0; n < N; n++)
    {
        for (size_t oc = 0; oc < Cout; oc++)
        {
            for (size_t oh = 0; oh < Hout; oh++)
            {
                for (size_t ow = 0; ow < Wout; ow++)
                {
                    const float grad =
                        dOut.At(n, oc, oh, ow);

                    for (size_t ic = 0; ic < Cin; ic++)
                    {
                        for (size_t kh = 0; kh < K; kh++)
                        {
                            for (size_t kw = 0; kw < K; kw++)
                            {
                                int ih = (int)(oh * Stride + kh - Padding);
                                int iw = (int)(ow * Stride + kw - Padding);

                                if (ih >= 0 && ih < (int)H && iw >= 0 && iw < (int)W)
                                    dInput.At(n, ic, ih, iw) += grad * Kernel.At(oc, ic, kh, kw);
                            }
                        }
                    }
                }
            }
        }
    }
}