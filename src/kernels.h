#pragma once
#include <stdexcept>
#include "tensor4d.h"
#include "tensor.h"
#include "thread_pool.h"



class Kernels
{
public:
	static void Multiply_Forward( Tensor& out, const Tensor& left, const Tensor& right);
	static void Multiply_Backward(Tensor& out, const Tensor& left, const Tensor& right);

    static void MatMul_Forward( Tensor& out, const Tensor& left, const Tensor& right);
    static void MatMul_Backward(Tensor& out, const Tensor& left, const Tensor& right);
    static inline void MatMul_Backward_A(Tensor& gradA, const Tensor& dOut, const Tensor& B);
    static inline void MatMul_Backward_B(Tensor& gradB, const Tensor& A, const Tensor& dOut);
    static inline void MatMul_Backward_A(Tensor4D& out, const Tensor4D& A, const Tensor4D& B);
    static inline void MatMul_Backward_B(Tensor4D& out, const Tensor4D& A, const Tensor4D& B);

    static void Tanh_Backward(Tensor& grad_in, const Tensor& out, const Tensor& outer_grad);

    static Tensor4D Softmax_CrossEntropy(const Tensor4D& logits, const Tensor4D& Target, Tensor4D& SoftmaxCache);
    static void Softmax_CrossEntropy_Backward(const Tensor4D& logits, Tensor4D& grad_logits, const Tensor4D& targets, const Tensor4D& outer);

    static void SGD_Update(Tensor& param, const Tensor& grad, Tensor& velocity, float lr, float momentum);
    static void SGD_Update(Tensor4D& param, const Tensor4D& grad, Tensor4D& velocity, float lr, float momentum);
    static void SGD_Update(Tensor4D& param, const Tensor4D& grad, Tensor4D& velocity, float lr, float momentum, float l2_decay);

    static void Conv2D_Forward(Tensor4D& out, const Tensor4D& Input, const Tensor4D& Kernel, int Stride = 1, int Padding = 0);
    static void Conv2D_Forward(ThreadPool& Pool, Tensor4D& out, const Tensor4D& Input, const Tensor4D& Kernel, int Stride, int Padding);
    static void Conv2D_Backward(Tensor4D& dInput, Tensor4D& dKernel, const Tensor4D& Input, const Tensor4D& Kernel, const Tensor4D& dOut, int Stride, int Padding);

    static Tensor4D MaxPool2D_Forward(const Tensor4D& Input, int KernelSize, int Stride, Tensor4D* ArgMax = nullptr);
    static void MaxPool2D_Backward(Tensor4D& dInput, const Tensor4D& Input, const Tensor4D& dOut, const Tensor4D& ArgMax, int KernelSize, int Stride);

    static void BatchNorm_Backward(  Tensor4D& dInput, const Tensor4D& Input, const Tensor4D& gradOut, const Tensor4D& mean, const Tensor4D& std);
    static void BatchNorm2D_Backward(Tensor4D& dInput, const Tensor4D& Input, const Tensor4D& gradOut, const Tensor4D& mean, const Tensor4D& std);
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



//
inline Tensor4D Kernels::Softmax_CrossEntropy(const Tensor4D& logits, const Tensor4D& Target, Tensor4D& SoftmaxCache)
{
    if (logits.GetBatches() != Target.GetBatches())
        throw std::runtime_error("Tensor4D Softmax_CrossEntropy(): size mismatch");
    if (logits.GetDepth() != 1)
        throw std::runtime_error("Tensor4D Softmax_CrossEntropy depth must be 1");

    // Target only contains the label (Sparse)
    if (Target.GetRows() != 1)
        throw std::runtime_error("Tensor4D CrossEntropy only support for sparse target");

    float loss = 0.0f;

    for (size_t b = 0; b < logits.GetBatches(); b++)
    {
        float max_val = -std::numeric_limits<float>::infinity();

        for (size_t r = 0; r < logits.GetRows(); r++)
            max_val = std::max(max_val, logits.At(b, 0, r, 0));

        float sum = 0.0f;

        for (size_t r = 0; r < logits.GetRows(); r++)
        {
            float val = std::exp(logits.At(b, 0, r, 0) - max_val);
            SoftmaxCache.At(b, 0, r, 0) = val;
            sum += val;
        }

        float inv_sum = 1.0f / sum;

        for (size_t r = 0; r < logits.GetRows(); r++)
            SoftmaxCache.At(b, 0, r, 0) *= inv_sum;

        float log_sum_exp = max_val + std::log(sum);

        int label = (int)Target.At(b, 0, 0, 0);

        if (label < 0 || label >= (int)logits.GetRows())
            throw std::runtime_error("Tensor4D Softmax_CrossEntropy invalid label index");

        loss += log_sum_exp - logits.At(b, 0, label, 0);
    }

    return Tensor4D({ 1 }, { loss / logits.GetBatches() });
}



//
inline void Kernels::Softmax_CrossEntropy_Backward(const Tensor4D& logits, Tensor4D& grad_logits, const Tensor4D& targets, const Tensor4D& outer)
{
    if (outer.GetBatches() != 1)
        throw std::runtime_error("Softmax_CrossEntropy_Backward outer must be scalar");

    const size_t N = logits.GetSize();
    const size_t B = logits.GetBatches();
    const size_t D = logits.GetDepth();

    for (size_t b = 0; b < B; b++)
        for (size_t d = 0; d < D; d++)
        {
            // ---- 1. max for numerical stability ----
            float max_val = -std::numeric_limits<float>::infinity();
            for (size_t i = 1; i < logits.GetRows(); i++)
            {
                float v = logits.At(b, d, i, 0);
                if (v > max_val) max_val = v;
            }

            // ---- 2. softmax ----
            float sum = 0.0f;
            for (size_t i = 0; i < logits.GetRows(); ++i)
            {
                float e = std::exp(logits.At(b, d, i, 0) - max_val);
                grad_logits.At(b, d, i, 0) = e; // reuse buffer temporarily
                sum += e;
            }

            // ---- 3. normalize ----
            float inv_sum = 1.0f / sum;

            for (size_t i = 0; i < logits.GetRows(); ++i)
                grad_logits.At(b, d, i, 0) *= inv_sum;

            // ---- 4. backward: (p - y) * outer_grad ----
            const int target = (int)targets.At(b, d, 0, 0);

            float og = outer.At(0, 0, 0, 0);

            for (size_t i = 0; i < logits.GetRows(); ++i)
            {
                float y = (i == (size_t)target) ? 1.0f : 0.0f;
                grad_logits.At(b, d, i, 0) = (grad_logits.At(b, d, i, 0) - y) * og;
            }
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



//out = A * B^T
inline void Kernels::MatMul_Backward_A(
    Tensor4D& out,
    const Tensor4D& A,
    const Tensor4D& B)
{
    const size_t outB = out.GetBatches();
    const size_t outD = out.GetDepth();

    const size_t I = A.GetRows();
    const size_t K = A.GetColumns();
    const size_t J = B.GetRows();   // B is logically transposed in forward

    for (size_t b = 0; b < outB; ++b)
        for (size_t d = 0; d < outD; ++d)
        {
            size_t aB = (A.GetBatches() == 1) ? 0 : b;
            size_t bB = (B.GetBatches() == 1) ? 0 : b;
            size_t aD = (A.GetDepth() == 1) ? 0 : d;
            size_t bD = (B.GetDepth() == 1) ? 0 : d;

            for (size_t i = 0; i < I; ++i)
                for (size_t j = 0; j < J; ++j)
                {
                    float sum = 0.0f;

                    // B^T: B[j,k]
                    for (size_t k = 0; k < K; ++k)
                    {
                        sum += A.At(aB, aD, i, k) *
                            B.At(bB, bD, j, k);
                    }

                    out.At(b, d, i, j) += sum;
                }
        }
}



//out = A^T * B
inline void Kernels::MatMul_Backward_B(
    Tensor4D& out,
    const Tensor4D& A,
    const Tensor4D& B)
{
    const size_t outB = out.GetBatches();
    const size_t outD = out.GetDepth();

    const size_t I = A.GetColumns(); // rows of A^T
    const size_t J = B.GetColumns();
    const size_t K = A.GetRows();

    for (size_t b = 0; b < outB; ++b)
        for (size_t d = 0; d < outD; ++d)
        {
            size_t aB = (A.GetBatches() == 1) ? 0 : b;
            size_t bB = (B.GetBatches() == 1) ? 0 : b;
            size_t aD = (A.GetDepth() == 1) ? 0 : d;
            size_t bD = (B.GetDepth() == 1) ? 0 : d;

            for (size_t i = 0; i < I; ++i)
                for (size_t j = 0; j < J; ++j)
                {
                    float sum = 0.0f;

                    for (size_t k = 0; k < K; ++k)
                    {
                        sum += A.At(aB, aD, k, i) *
                            B.At(bB, bD, k, j);
                    }

                    out.At(b, d, i, j) += sum;
                }
        }
}



inline void Kernels::SGD_Update(Tensor& param, const Tensor& grad, Tensor& velocity, float lr, float momentum)
{
    float* p = param.Data();
    const float* g = grad.Data();
    float* v = velocity.Data();

    const size_t n = param.Size();

    for (size_t i = 0; i < n; i++)
    {
        v[i] = momentum * v[i] - lr * g[i];
        p[i] += v[i];
    }
}



inline void Kernels::SGD_Update(Tensor4D& param, const Tensor4D& grad, Tensor4D& velocity, float lr, float momentum)
{
    float* p = param.Data();
    const float* g = grad.Data();
    float* v = velocity.Data();

    const size_t n = param.GetSize();

    for (size_t i = 0; i < n; i++)
    {
        v[i] = momentum * v[i] - lr * g[i];
        p[i] += v[i];
    }
}



inline void Kernels::SGD_Update(Tensor4D& param, const Tensor4D& grad, Tensor4D& velocity, float lr, float momentum, float l2_decay)
{
    float* p = param.Data();
    const float* g = grad.Data();
    float* v = velocity.Data();

    const size_t n = param.GetSize();

    for (size_t i = 0; i < n; i++)
    {
        v[i] = momentum * v[i] - lr * (g[i] + l2_decay * p[i]);
        p[i] += v[i];
    }
}



inline void Kernels::Conv2D_Forward(Tensor4D& out, const Tensor4D& Input, const Tensor4D& Kernel, int Stride, int Padding)
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

    if (out.GetShape()[0] != N)
		throw std::runtime_error("Conv2D_Forward(): Output shape mismatch");
    if (out.GetShape()[1] != out_channels)
        throw std::runtime_error("Conv2D_Forward(): Output shape mismatch");
    if (out.GetShape()[2] != H_out)
        throw std::runtime_error("Conv2D_Forward(): Output shape mismatch");
    if (out.GetShape()[3] != W_out)
        throw std::runtime_error("Conv2D_Forward(): Output shape mismatch");

    // =========================================================
    // FAST PATH: all tensors contiguous
    // =========================================================

    if (Input.IsContiguous() && Kernel.IsContiguous() && out.IsContiguous())
    {
        float* __restrict outPtr = out.Data();
        const float* __restrict inPtr = Input.Data();
        const float* __restrict kPtr = Kernel.Data();

        const size_t in_HW = H * W;
        const size_t out_HW = H_out * W_out;
        const size_t k_HW = kernel_size * kernel_size;

        for (size_t n = 0; n < N; n++)
        {
            const float* in_n = inPtr + n * C_in * in_HW;
            float* out_n = outPtr + n * out_channels * out_HW;

            for (size_t oc = 0; oc < out_channels; oc++)
            {
                const float* k_oc = kPtr + oc * C_in * k_HW;
                float* out_oc = out_n + oc * out_HW;

                for (size_t oh = 0; oh < H_out; oh++)
                {
                    for (size_t ow = 0; ow < W_out; ow++)
                    {
                        float sum = 0.0f;

                        const int ih_base = (int)oh * Stride - Padding;
                        const int iw_base = (int)ow * Stride - Padding;

                        for (size_t ic = 0; ic < C_in; ic++)
                        {
                            const float* in_ic = in_n + ic * in_HW;
                            const float* k_ic = k_oc + ic * k_HW;

                            for (size_t kh = 0; kh < kernel_size; kh++)
                            {
                                const int ih = ih_base + (int)kh;
                                if (ih < 0 || ih >= (int)H) continue;

                                const float* in_row = in_ic + ih * W;

                                for (size_t kw = 0; kw < kernel_size; kw++)
                                {
                                    const int iw = iw_base + (int)kw;
                                    if (iw < 0 || iw >= (int)W) continue;

                                    sum += in_row[iw] * k_ic[kh * kernel_size + kw];
                                }
                            }
                        }

                        out_oc[oh * W_out + ow] = sum;
                    }
                }
            }
        }

        return;
    }

    // =========================================================
    // SLOW PATH
    // =========================================================

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

                    out.At(n, oc, oh, ow) = sum;
                }
            }
        }
    }
}



inline void Kernels::Conv2D_Forward(ThreadPool& Pool, Tensor4D& out, const Tensor4D& Input, const Tensor4D& Kernel, int Stride, int Padding)
{
    // Output shape berechnen
    size_t N = Input.GetShape()[0];
    size_t C_in = Input.GetShape()[1];
    size_t H = Input.GetShape()[2];
    size_t W = Input.GetShape()[3];

    size_t out_channels = Kernel.GetShape()[0];
    size_t in_channels = Kernel.GetShape()[1];
    size_t kernel_size = Kernel.GetShape()[2];

    size_t H_out = (H + 2 * Padding - kernel_size) / Stride + 1;
    size_t W_out = (W + 2 * Padding - kernel_size) / Stride + 1;

    if (out.GetShape()[0] != N)
        throw std::runtime_error("Conv2D_Forward(): Output shape mismatch");
    if (out.GetShape()[1] != out_channels)
        throw std::runtime_error("Conv2D_Forward(): Output shape mismatch");
    if (out.GetShape()[2] != H_out)
        throw std::runtime_error("Conv2D_Forward(): Output shape mismatch");
    if (out.GetShape()[3] != W_out)
        throw std::runtime_error("Conv2D_Forward(): Output shape mismatch");

    // =========================================================
    // FAST PATH: all tensors contiguous
    // =========================================================

    if (Input.IsContiguous() && Kernel.IsContiguous())
    {
        float* __restrict outPtr = out.Data();
        const float* __restrict inPtr = Input.Data();
        const float* __restrict kPtr = Kernel.Data();

        const size_t in_HW = H * W;
        const size_t out_HW = H_out * W_out;
        const size_t k_HW = kernel_size * kernel_size;

        size_t chunk = N / Pool.GetWorkerCount();
        if (chunk == 0)
            chunk = 1;

        for (size_t n0 = 0; n0 < N; n0 += chunk)
        {
            size_t n1 = std::min(N, n0 + chunk);

            Pool.Enqueue([=]()
            {
                for (size_t n = n0; n < n1; n++)
                {
                    const float* in_n = inPtr + n * C_in * in_HW;
                    float* out_n = outPtr + n * out_channels * out_HW;

                    for (size_t oc = 0; oc < out_channels; oc++)
                    {
                        const float* k_oc = kPtr + oc * C_in * k_HW;
                        float* out_oc = out_n + oc * out_HW;

                        for (size_t oh = 0; oh < H_out; oh++)
                        {
                            for (size_t ow = 0; ow < W_out; ow++)
                            {
                                float sum = 0.0f;

                                const int ih_base = (int)oh * Stride - Padding;
                                const int iw_base = (int)ow * Stride - Padding;

                                for (size_t ic = 0; ic < C_in; ic++)
                                {
                                    const float* in_ic = in_n + ic * in_HW;
                                    const float* k_ic = k_oc + ic * k_HW;

                                    for (size_t kh = 0; kh < kernel_size; kh++)
                                    {
                                        const int ih = ih_base + (int)kh;
                                        if (ih < 0 || ih >= (int)H) continue;

                                        const float* in_row = in_ic + ih * W;

                                        for (size_t kw = 0; kw < kernel_size; kw++)
                                        {
                                            const int iw = iw_base + (int)kw;
                                            if (iw < 0 || iw >= (int)W) continue;

                                            sum += in_row[iw] * k_ic[kh * kernel_size + kw];
                                        }
                                    }
                                }

                                out_oc[oh * W_out + ow] = sum;
                            }
                        }
                    }
                }
            });
        }

        Pool.Wait();
        return;
    }

    // =========================================================
    // SLOW PATH
    // =========================================================

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

                    out.At(n, oc, oh, ow) = sum;
                }
            }
        }
    }
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



inline Tensor4D Kernels::MaxPool2D_Forward(const Tensor4D& Input, int KernelSize, int Stride, Tensor4D* ArgMax)
{
    size_t N = Input.GetShape()[0];
    size_t C = Input.GetShape()[1];
    size_t H = Input.GetShape()[2];
    size_t W = Input.GetShape()[3];

    size_t H_out = (H - KernelSize) / Stride + 1;
    size_t W_out = (W - KernelSize) / Stride + 1;

    Tensor4D Output({ N, C, H_out, W_out });

    for (size_t n = 0; n < N; n++)
    {
        for (size_t c = 0; c < C; c++)
        {
            for (size_t oh = 0; oh < H_out; oh++)
            {
                for (size_t ow = 0; ow < W_out; ow++)
                {
                    float maxVal = -FLT_MAX;
                    size_t maxIndex = 0;

                    for (size_t kh = 0; kh < KernelSize; kh++)
                    {
                        for (size_t kw = 0; kw < KernelSize; kw++)
                        {
                            size_t ih = oh * Stride + kh;
                            size_t iw = ow * Stride + kw;

                            float v = Input.At(n, c, ih, iw);

                            if (v > maxVal)
                            {
                                maxVal = v;
                                maxIndex = ih * W + iw; // flatten index
                            }
                        }
                    }

                    Output.At(n, c, oh, ow) = maxVal;
                    if (ArgMax)
                        ArgMax->At(n, c, oh, ow) = (float)maxIndex;
                }
            }
        }
    }

    return Output;
}



inline void Kernels::MaxPool2D_Backward(Tensor4D& dInput, const Tensor4D& Input, const Tensor4D& dOut, const Tensor4D& ArgMax, int KernelSize, int Stride)
{
    dInput.SetZero();

    const size_t N = Input.GetShape()[0];
    const size_t C = Input.GetShape()[1];
    const size_t H = Input.GetShape()[2];
    const size_t W = Input.GetShape()[3];

    const size_t Hout = dOut.GetShape()[2];
    const size_t Wout = dOut.GetShape()[3];

    for (size_t n = 0; n < N; n++)
    {
        for (size_t c = 0; c < C; c++)
        {
            for (size_t oh = 0; oh < Hout; oh++)
            {
                for (size_t ow = 0; ow < Wout; ow++)
                {
                    float grad = dOut.At(n, c, oh, ow);

                    // ArgMax ist flattened index
                    size_t idx = (size_t)ArgMax.At(n, c, oh, ow);

                    size_t ih = idx / W;
                    size_t iw = idx % W;

                    // safety check (optional)
                    if (ih < H && iw < W)
                        dInput.At(n, c, ih, iw) += grad;
                }
            }
        }
    }
}



inline void Kernels::BatchNorm_Backward(Tensor4D& dInput, const Tensor4D& Input, const Tensor4D& gradOut, const Tensor4D& mean, const Tensor4D& std)
{
    const float eps = 0.001f;

    const size_t N = dInput.GetBatches();

    std::vector<float> dxmu(N);

    for (size_t d = 0; d < dInput.GetDepth(); d++)
    {
        for (size_t r = 0; r < dInput.GetRows(); r++)
        {
            for (size_t c = 0; c < dInput.GetColumns(); c++)
            {
                float mu = mean.At(0, d, r, c);
                float sigma = std.At(0, d, r, c);

                float invStd = 1.0f / (sigma + eps);

                //----------------------------------------
                // dStd
                //----------------------------------------

                float dStd = 0.0f;

                for (size_t b = 0; b < N; b++)
                {
                    float xmu = Input.At(b, d, r, c) - mu;

                    dStd += gradOut.At(b, d, r, c) * (-xmu) * invStd * invStd;
                }

                //----------------------------------------
                // dVar
                //----------------------------------------

                float dVar = dStd * 0.5f / (sigma + eps);

                //----------------------------------------
                // dxmu
                //----------------------------------------

                for (size_t b = 0; b < N; b++)
                {
                    float xmu = Input.At(b, d, r, c) - mu;

                    dxmu[b] = gradOut.At(b, d, r, c) * invStd;

                    dxmu[b] += dVar * 2.0f * xmu / (float)N;
                }

                //----------------------------------------
                // dMean
                //----------------------------------------

                float dMean = 0.0f;

                for (size_t b = 0; b < N; b++)
                    dMean -= dxmu[b];

                //----------------------------------------
                // dX
                //----------------------------------------

                for (size_t b = 0; b < N; b++)
                {
                    dInput.At(b, d, r, c) = dxmu[b] + dMean / (float)N;
                }
            }
        }
    }
}



inline void Kernels::BatchNorm2D_Backward(Tensor4D& dInput, const Tensor4D& Input, const Tensor4D& gradOut, const Tensor4D& mean, const Tensor4D& std)
{
    const float eps = 0.001f;

    const size_t N =
        dInput.GetBatches() *
        dInput.GetRows() *
        dInput.GetColumns();

    std::vector<float> dxmu(N);

    for (size_t d = 0; d < dInput.GetDepth(); d++)
    {
        float mu = mean.At(0, d, 0, 0);
        float sigma = std.At(0, d, 0, 0);

        float invStd = 1.0f / (sigma + eps);

        //----------------------------------------
        // dStd
        //----------------------------------------

        float dStd = 0.0f;

        for (size_t b = 0; b < dInput.GetBatches(); b++)
        {
            for (size_t r = 0; r < dInput.GetRows(); r++)
            {
                for (size_t c = 0; c < dInput.GetColumns(); c++)
                {
                    float xmu =
                        Input.At(b, d, r, c) - mu;

                    dStd +=
                        gradOut.At(b, d, r, c)
                        * (-xmu)
                        * invStd
                        * invStd;
                }
            }
        }

        //----------------------------------------
        // dVar
        //----------------------------------------

        float dVar =
            dStd * 0.5f / (sigma + eps);

        //----------------------------------------
        // dxmu
        //----------------------------------------

        size_t idx = 0;

        for (size_t b = 0; b < dInput.GetBatches(); b++)
        {
            for (size_t r = 0; r < dInput.GetRows(); r++)
            {
                for (size_t c = 0; c < dInput.GetColumns(); c++)
                {
                    float xmu =
                        Input.At(b, d, r, c) - mu;

                    dxmu[idx] =
                        gradOut.At(b, d, r, c)
                        * invStd;

                    dxmu[idx] +=
                        dVar
                        * 2.0f
                        * xmu
                        / (float)N;

                    idx++;
                }
            }
        }

        //----------------------------------------
        // dMean
        //----------------------------------------

        float dMean = 0.0f;

        for (float v : dxmu)
            dMean -= v;

        //----------------------------------------
        // dX
        //----------------------------------------

        idx = 0;

        for (size_t b = 0; b < dInput.GetBatches(); b++)
        {
            for (size_t r = 0; r < dInput.GetRows(); r++)
            {
                for (size_t c = 0; c < dInput.GetColumns(); c++)
                {
                    dInput.At(b, d, r, c) =
                        dxmu[idx]
                        + dMean / (float)N;

                    idx++;
                }
            }
        }
    }
}