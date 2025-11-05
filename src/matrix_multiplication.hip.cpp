#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "hip/hip_runtime.h"

__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    if(row < M && col < K){
        float sum = 0.0f;
        for(int i = 0; i < N; i++){
            sum += A[row * N + i] * B[i * K + col];
        }
        C[row * K + col] = sum;
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
int main(void) {
    int N, M, K;

    float *A, *B, *C;
    float *d_A, *d_B, *d_C;

    scanf("%d", &M);
    scanf("%d", &N);
    scanf("%d", &K);

    A = (float*)malloc(M * N * sizeof(float));
    B = (float*)malloc(N * K * sizeof(float));
    C = (float*)malloc(M * K * sizeof(float));

    for(int i = 0; i < M; i++)
        for(int j = 0; j < N; j++)
            scanf("%f", &A[i * N + j]);


    for(int i = 0; i < N; i++)
        for(int j = 0; j < K; j++)
            scanf("%f", &B[i * K + j]);


    hipMalloc(&d_A, M * N * sizeof(float));
    hipMalloc(&d_B, N * K * sizeof(float));
    hipMalloc(&d_C, M * K * sizeof(float));

    hipMemcpy(d_A, A, M * N * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_B, B, N * K * sizeof(float), hipMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
            (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, M, N, K);
    hipDeviceSynchronize();

    hipMemcpy(C, d_C, M * K * sizeof(float), hipMemcpyDeviceToHost);

    for(int i = 0; i < M; i++){
        for(int j = 0; j < K; j++)
            printf("%.1f ", C[i * K + j]);
        printf("\n");
    }

    free(A);
    free(B);
    free(C);
    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);
}


