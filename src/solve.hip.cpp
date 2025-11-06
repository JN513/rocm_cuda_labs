#include <stdio.h>
#include <stdlib.h>
#include <hip/hip_runtime.h>

__global__ void reduce_sum(const float* input, float* partial, int N) {
    __shared__ float cache[256];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int local = threadIdx.x;
    float sum = 0.0f;

    // cada thread soma vários elementos
    for (int i = tid; i < N; i += blockDim.x * gridDim.x)
        sum += input[i];

    cache[local] = sum;
    __syncthreads();

    // redução dentro do bloco
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (local < stride)
            cache[local] += cache[local + stride];
        __syncthreads();
    }

    // thread 0 escreve resultado parcial
    if (local == 0)
        partial[blockIdx.x] = cache[0];
}

extern "C" void solve(const float* input, float* output, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    float* d_input;
    float* d_partial;

    hipMalloc(&d_input, N * sizeof(float));
    hipMalloc(&d_partial, blocks * sizeof(float));

    hipMemcpy(d_input, input, N * sizeof(float), hipMemcpyHostToDevice);

    // 1ª fase: soma parcial por bloco
    reduce_sum<<<blocks, threads>>>(d_input, d_partial, N);
    hipDeviceSynchronize();

    // 2ª fase: soma final no host
    float* h_partial = (float*)malloc(blocks * sizeof(float));
    hipMemcpy(h_partial, d_partial, blocks * sizeof(float), hipMemcpyDeviceToHost);

    float total = 0.0f;
    for (int i = 0; i < blocks; i++)
        total += h_partial[i];

    output[0] = total;  // grava resultado final

    free(h_partial);
    hipFree(d_input);
    hipFree(d_partial);
}

int main(){
    int N;

    scanf("%d", &N);

    float *A = (float *)malloc(N * sizeof(float));
    float B;

    for(int i = 0; i < N; i++)
        scanf("%f", &A[i]);

    solve(A, &B, N);

    printf("Sum = %.1f\n", B);

    free(A);
}