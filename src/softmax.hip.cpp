#include <stdio.h>
#include <hip/hip_runtime.h>

__global__ void softmax_kernel(const float* input, float* output, int N) {
    extern __shared__ float shmem[];
    int tid = threadIdx.x;

    // Copiar entrada para shared memory
    float x = -INFINITY;
    if (tid < N)
        x = input[tid];
    shmem[tid] = x;
    __syncthreads();

    // Redução para achar o máximo
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride && (tid + stride) < N)
            shmem[tid] = fmaxf(shmem[tid], shmem[tid + stride]);
        __syncthreads();
    }
    float max_val = shmem[0];
    __syncthreads();

    // Calcular exp(x - max)
    float exp_val = 0.0f;
    if (tid < N) {
        exp_val = expf(x - max_val);
        shmem[tid] = exp_val;
    }
    __syncthreads();

    //  Redução para soma total
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride && (tid + stride) < N)
            shmem[tid] += shmem[tid + stride];
        __syncthreads();
    }
    float sum_exp = shmem[0];
    __syncthreads();

    // Normalização
    if (tid < N)
        output[tid] = exp_val / sum_exp;
}