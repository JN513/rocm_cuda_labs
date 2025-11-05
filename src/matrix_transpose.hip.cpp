#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "hip/hip_runtime.h"


__global__ void matrix_transpose_kernel(const float* input, float* output, int rows, int cols) {
    int col = blockDim.y * blockIdx.y + threadIdx.y;
    int row = blockDim.x * blockIdx.x + threadIdx.x;

    if(col < cols && row < rows) {
        output[row * cols + col] = input[col * rows + row];
        output[col * rows + row] = input[row * cols + col];
    }
}


// A, B, C are device pointers (i.e. pointers to memory on the GPU)
int main(void) {
    int N, M;

    float *A, *B;
    float *d_A, *d_B;

    scanf("%d", &M);
    scanf("%d", &N);

    A = (float*)malloc(M * N * sizeof(float));
    B = (float*)malloc(N * M * sizeof(float));

    for(int i = 0; i < M; i++)
        for(int j = 0; j < N; j++)
            scanf("%f", &A[i * N + j]);



    hipMalloc(&d_A, M * N * sizeof(float));
    hipMalloc(&d_B, N * M * sizeof(float));

    hipMemcpy(d_A, A, M * N * sizeof(float), hipMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
            (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrix_transpose_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, M, N);
    hipDeviceSynchronize();

    hipMemcpy(B, d_B, M * N * sizeof(float), hipMemcpyDeviceToHost);

    for(int i = 0; i < N; i++){
        for(int j = 0; j < M; j++)
            printf("%.1f ", B[i * N + j]);
        printf("\n");
    }

    free(A);
    free(B);
    hipFree(d_A);
    hipFree(d_B);
}


