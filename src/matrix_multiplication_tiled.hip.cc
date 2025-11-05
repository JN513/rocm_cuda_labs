#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "hip/hip_runtime.h"


#define TILE_SIZE 32

#define TILE_SIZE 16

__global__ void matmul(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;

    // Loop sobre as tiles ao longo da dimensão N (colunas de A = linhas de B)
    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Carrega tile de A (dimensões MxN)
        if (row < M && t * TILE_SIZE + threadIdx.x < N)
            tileA[threadIdx.y][threadIdx.x] = A[row * N + t * TILE_SIZE + threadIdx.x];
        else
            tileA[threadIdx.y][threadIdx.x] = 0.0f;

        // Carrega tile de B (dimensões NxK)
        if (col < K && t * TILE_SIZE + threadIdx.y < N)
            tileB[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * K + col];
        else
            tileB[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // Multiplica os tiles carregados
        for (int j = 0; j < TILE_SIZE; j++)
            sum += tileA[threadIdx.y][j] * tileB[j][threadIdx.x];

        __syncthreads();
    }

    // Escreve o resultado final na matriz C (dimensões MxK)
    if (row < M && col < K)
        C[row * K + col] = sum;
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

    matmul<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, M, N, K);
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