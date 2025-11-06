#include <stdio.h>
#include <stdlib.h>
#include <hip/hip_runtime.h>


#define BLOCK_SIZE 256

__global__ void reverse_array(float* input, int N) {
    __shared__ float tile[BLOCK_SIZE];

    int start = blockIdx.x * blockDim.x;
    int end   = N - start - 1;
    int tid   = threadIdx.x;

    if (start + tid < N)
        tile[tid] = input[start + tid];

    __syncthreads();

    if (start + tid < N)
        input[start + tid] = input[end - tid];
}


int main(){
    int N;

    scanf("%d", &N);

    float *A = (float *)malloc(N * sizeof(float));

    for(int i = 0; i < N; i++)
        scanf("%f", &A[i]);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    float *d_A;

    hipMalloc(&d_A, sizeof(float) * N);

    hipMemcpy(d_A, A, sizeof(float) * N, hipMemcpyHostToDevice);

    reverse_array <<<blocksPerGrid, threadsPerBlock>>> (d_A, N);

    hipDeviceSynchronize();

    hipMemcpy(A, d_A, sizeof(float) * N, hipMemcpyDeviceToHost);

    for(int i = 0; i < N; i++)
        printf("%.1f ", A[i]);

    printf("\n");

    free(A);
    hipFree(d_A);

}