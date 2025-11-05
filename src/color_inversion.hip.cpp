#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "hip/hip_runtime.h"


__global__ void invert_kernel(unsigned char* image, int width, int height) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if(idx < width * height * 4 && (idx & 3) != 3){
        image[idx] = 255 - image[idx];
    }
}
// A, B, C are device pointers (i.e. pointers to memory on the GPU)
int main(void) {
    int N, M;

    unsigned char *image;
    unsigned char *d_image;

    scanf("%d", &N);
    scanf("%d", &M);

    image = (unsigned char *)malloc(N * M * sizeof(char));

    for(int i = 0; i < N * M * 4; i++){
        int temp;
        scanf("%d", &temp);
        image[i] = (unsigned char)temp;
    }


    hipMalloc(&d_image, N * M * 4 * sizeof(char));

    hipMemcpy(d_image, image, 4 * N * M * sizeof(char), hipMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N * M + threadsPerBlock - 1) / threadsPerBlock;

    invert_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_image, N, M);
    hipDeviceSynchronize();

    hipMemcpy(image, d_image, 4 * N * M * sizeof(char), hipMemcpyDeviceToHost);

    for(int i = 0; i < 4 * N * M; i++)
        printf("%d ", image[i]);
    printf("\n");

    free(image);
    hipFree(d_image);
}
