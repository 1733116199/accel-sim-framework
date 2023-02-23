#include <cuda_runtime.h>
#include <stdio.h>
#include <cassert>

// Each thread block is maxed out (1024)
#define NT 1024
// Number of thread blocks 2^20
// #define NB 1048756ull
#define NB 1048
// The data is large enough such that each thread
// works on one float
#define N (NB * NT)
// This controls the arithmetic intensity
#define NUM_OPS 7000

// Common macro for CUDA applications
#define gpuErrorCheck(ans, abort)                    \
    {                                                \
        gpuAssert((ans), __FILE__, __LINE__, abort); \
    }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "assert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
        {
            exit(code);
        }
    }
}

/**
 * This is a kernel that does "num_ops" MAC for every 2 memory accesses
 * Arithmetic intensity = num_ops (FLOPS per float accessed)
 */
__global__ void measure_max_flops(float *a)
{
    // uint i = blockIdx.x * NT + threadIdx.x;
    uint i;
    asm("mad.lo.u32 %0, %1, %2, %3;"
        : "=r"(i)
        : "r"(blockIdx.x), "n"(NT), "r"(threadIdx.x));

    // float * addr = &a[i];
    float *addr;
    asm("mad.wide.u32 %0, %1, %2, %3;"
        : "=l"(addr)
        : "r"(i), "n"(sizeof(float)), "l"(a));

    float input = *addr;
    float output = 0;

    for (int j = 0; j < NUM_OPS; j++)
    {
        // Each iteration is a MAC op
        // Use inline assembly here to prevent compiler optimization
        // Equivalent C code: output += input * input;
        asm("fma.rn.f32 %0, %1, %1, %0;"
            : "+f"(output)
            : "f"(input));
    }
    *addr = output;
}

// common macro for CUDA kernels
#define ALLOC_FLOAT_GPU(SYMBOL, NUM, INIT_CODE)                                                                \
    float *SYMBOL;                                                                                             \
    float *SYMBOL##_device;                                                                                    \
    {                                                                                                          \
        SYMBOL = (float *)calloc(NUM, sizeof(float));                                                          \
        assert(SYMBOL != NULL);                                                                                \
        INIT_CODE;                                                                                             \
        gpuErrorCheck(cudaMalloc(&SYMBOL##_device, sizeof(float) * NUM), true);                                \
        gpuErrorCheck(cudaMemcpy(SYMBOL##_device, SYMBOL, sizeof(float) * NUM, cudaMemcpyHostToDevice), true); \
    }

// common macro for CUDA kernels
#define TIME(CODE, SYMBOL)                          \
    float SYMBOL;                                   \
    {                                               \
        cudaEvent_t start, stop;                    \
        cudaEventCreate(&start);                    \
        cudaEventCreate(&stop);                     \
        cudaEventRecord(start);                     \
        CODE;                                       \
        cudaDeviceSynchronize();                    \
        cudaEventRecord(stop);                      \
        cudaEventSynchronize(stop);                 \
        cudaEventElapsedTime(&SYMBOL, start, stop); \
    }

int main()
{
    ALLOC_FLOAT_GPU(a, N, {
        for (unsigned long long i = 0; i < N; i++)
        {
            a[i] = i;
        }
    });

    // launch and time the kernel
    TIME((measure_max_flops<<<NB, NT>>>(a_device)), ms);

    gpuErrorCheck(cudaMemcpy(a, a_device, sizeof(float) * N, cudaMemcpyDeviceToHost), true);

    for (int i = 0; i < 20; i++)
    {
        // printf("Actual: %.0f Expected: %.0f\n", a[i], (float)i * i * NUM_OPS);
        if (i < 20)
        {
            assert((i * i * NUM_OPS) == (unsigned long long)a[i]);
        }
    }

    double gflops = NB * NT * NUM_OPS * 2 / (ms / 1000.0) / 1000000000.0;
    double ai = NUM_OPS / 2.0 / sizeof(float);
    double bw = NB * NT * 2 * sizeof(float) / (ms / 1000.0) / 1000000000;

    printf("Time Elapsed: %.4f ms\n", ms);
    printf("Arithmetic Intensity (flops per byte): %.4f\n", ai);
    printf("Effective bandwidth (GB per second): %.4f\n", bw);
    printf("GFLOPS Measured: %.2f\n", gflops);
}