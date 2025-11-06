#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "hip/hip_runtime.h"

__constant__ float d_kernel[2048];

__global__ void convolution_1d_kernel(const float* input, float* output,
                                      int input_size, int kernel_size) {
    int output_size = input_size - kernel_size + 1;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < output_size) {
        float sum = 0.0f;

        #pragma unroll
        for (int k = 0; k < kernel_size; k++) {
            sum += input[i + k] * d_kernel[k];
        }
        output[i] = sum;
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
int main(void) {
    int N, M;

    float *A, *B, *C;
    float *d_A, *d_B, *d_C;

    scanf("%d", &N);
    scanf("%d", &M);

    A = (float *)malloc(N * sizeof(float));
    B = (float *)malloc(M * sizeof(float));
    C = (float *)malloc((N - M + 1) * sizeof(float));

    for(int i = 0; i < N; i++)
        scanf("%f", &A[i]);

    for(int i = 0; i < M; i++)
        scanf("%f", &B[i]);

    hipMalloc(&d_A, N * sizeof(float));
    //hipMalloc(&d_B, M * sizeof(float));
    hipMalloc(&d_C, (N - M + 1) * sizeof(float));

    hipMemcpy(d_A, A, N * sizeof(float), hipMemcpyHostToDevice);
    //hipMemcpy(d_B, B, M * sizeof(float), hipMemcpyHostToDevice);

    hipMemcpyToSymbol(d_kernel, B, M * sizeof(float));

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    convolution_1d_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_C, N, M);
    hipDeviceSynchronize();

    hipMemcpy(C, d_C, (N - M + 1) * sizeof(float), hipMemcpyDeviceToHost);

    for(int i = 0; i < (N - M + 1); i++)
        printf("%.1f ", C[i]);
    printf("\n");

    free(A);
    free(B);
    free(C);
    hipFree(d_A);
    //hipFree(d_B);
    hipFree(d_C);
}
