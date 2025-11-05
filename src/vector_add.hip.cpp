#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "hip/hip_runtime.h"


__global__ void vector_add(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < N)
        C[idx] = A[idx] + B[idx];
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
int main(void) {
    int N;

    float *A, *B, *C;
    float *d_A, *d_B, *d_C;

    scanf("%d", &N);

    A = (float*)malloc(N * sizeof(float));
    B = (float*)malloc(N * sizeof(float));
    C = (float*)malloc(N * sizeof(float));

    for(int i = 0; i < N; i++)
        scanf("%f", &A[i]);

    for(int i = 0; i < N; i++)
        scanf("%f", &B[i]);

    size_t size = N * sizeof(float);

    hipMalloc(&d_A, size);
    hipMalloc(&d_B, size);
    hipMalloc(&d_C, size);

    hipMemcpy(d_A, A, size, hipMemcpyHostToDevice);
    hipMemcpy(d_B, B, size, hipMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    vector_add<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    hipDeviceSynchronize();

    hipMemcpy(C, d_C, size, hipMemcpyDeviceToHost);

    for(int i = 0; i < N; i++)
        printf("%.1f ", C[i]);
    printf("\n");

    free(A);
    free(B);
    free(C);
    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);
}
