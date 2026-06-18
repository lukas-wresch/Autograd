#include <chrono>
#include <iostream>
#define NOMINMAX
#include <windows.h>
#include "../src/kernels.h"



double RunConvBenchmark(size_t N, size_t C, size_t H, size_t W, int threads)
{
    Tensor4D input({ N, C, H, W });
    Tensor4D kernel({ 16, C, 3, 3 });   // 16 Output-Channels, 3x3 Kernel
    Tensor4D out({ N, 16, (H - 2), (W - 2) });

    // deterministic init
    for (size_t i = 0; i < input.GetSize(); i++)
        input.Data()[i] = 1.0f;

    for (size_t i = 0; i < kernel.GetSize(); i++)
        kernel.Data()[i] = 0.01f;

    ThreadPool pool(threads);

    // Warmup
    Kernels::Conv2D_Forward(pool, out, input, kernel, 1, 0);

    auto start = std::chrono::high_resolution_clock::now();

    Kernels::Conv2D_Forward(pool, out, input, kernel, 1, 0);

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> ms = end - start;
    return ms.count();
}



double RunConvBackwardBenchmark(size_t N, size_t C, size_t H, size_t W, int threads)
{
    Tensor4D input({ N, C, H, W });
    Tensor4D dinput({ N, C, H, W });
    Tensor4D kernel({ 16, C, 3, 3 });   // 16 Output-Channels, 3x3 Kernel
    Tensor4D dkernel({ 16, C, 3, 3 });   // 16 Output-Channels, 3x3 Kernel
    Tensor4D out({ N, 16, (H - 2), (W - 2) });

    // deterministic init
    for (size_t i = 0; i < input.GetSize(); i++)
        input.Data()[i] = 1.0f;

    for (size_t i = 0; i < kernel.GetSize(); i++)
        kernel.Data()[i] = 0.01f;

    ThreadPool pool(threads);

    // Warmup
    Kernels::Conv2D_Backward(pool, dinput, dkernel, input, kernel, out, 1, 0);

    auto start = std::chrono::high_resolution_clock::now();

    Kernels::Conv2D_Backward(pool, dinput, dkernel, input, kernel, out, 1, 0);

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> ms = end - start;
    return ms.count();
}



int main()
{
    std::vector<int> threadCounts = { 1, 2, 4, 8, 16, 32 };
    std::vector<int> BatchCounts  = { 1, 4, 16, 64 };

    std::vector<size_t> sizes =
    {
        8,
        16,
        32,
        64,
        128,
        256,
        //512,
    };

    printf("\n==============================\n");
    printf(" Conv2D Benchmark (3x3 Kernel)\n");
    printf("==============================\n");

    for (size_t size : sizes)
        for (size_t b : BatchCounts)
    {
        std::stringstream ss;
        ss << "\nInput: " << b << "x" << 16 << "x" << size << "x" << size << "\n";

        std::cout << ss.str();

        double baseline = 0.0;

        for (int threads : threadCounts)
        {
            //double ms = RunConvBenchmark(
            double ms = RunConvBackwardBenchmark(
                b,      // N
                16,     // C
                size,
                size,
                threads);

            if (threads == 1)
                baseline = ms;

            double speedup = baseline / ms;

            std::stringstream line;

            line << "Threads=" << threads << "  Time=" << ms << " ms";

            if (threads > 1)
                line << "  Speedup=" << speedup << "x";

            line << "\n";

            std::cout << line.str() << std::endl;
        }
    }

    return 0;
}