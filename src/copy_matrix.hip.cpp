#include <stdio.h>
#include <stdlib.h>
#include <hip/hip_runtime.h>

__global__ void copy_matrix_kernel(const float* A, float* B, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx < N * N){
        B[idx] = A[idx];
    }
}

int main(){
    int N;

    scanf("%d", &N);

    int total_elements = N * N;

    float *A = (float *)malloc(total_elements * sizeof(float));
    float *B = (float *)malloc(total_elements * sizeof(float));

    for(int i = 0; i < total_elements; i++)
        scanf("%f", &A[i]);

    int threadsPerBlock = 256;
    int blocksPerGrid = (total_elements + threadsPerBlock - 1) / threadsPerBlock;

    float *d_A, *d_B;

    hipMalloc(&d_A, sizeof(float) * total_elements);
    hipMalloc(&d_B, sizeof(float) * total_elements);

    hipMemcpy(d_A, A, sizeof(float) * total_elements, hipMemcpyHostToDevice);

    copy_matrix_kernel <<<blocksPerGrid, threadsPerBlock>>> (d_A, d_B, N);

    hipDeviceSynchronize();

    hipMemcpy(A, d_B, sizeof(float) * total_elements, hipMemcpyDeviceToHost);

    for(int i = 0; i < total_elements; i++)
        printf("%.1f ", A[i]);

    printf("\n");

    free(A);
    hipFree(d_A);

}