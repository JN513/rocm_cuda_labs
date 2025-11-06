#include <stdio.h>
#include <hip/hip_runtime.h>

__global__ void test_norm(float2 v, float* result) {
    // Computa a norma euclidiana: sqrt(x^2 + y^2)
    // O compilador otimiza isso para a instrução PTX `norm.f32`
    //*result = norm(v.x, v.y);
    double temp[] = {v.x, v.y, 0.0};
    //*result = norm(2, temp);
    *result = norm3d(v.x, v.y, 0.0);
    //*result = sqrtf(fmaf(v.x, v.x, v.y * v.y));
}

int main() {
    float2 v = make_float2(3.0f, 4.0f);
    float* d_result;
    float h_result;

    hipMalloc(&d_result, sizeof(float));
    test_norm<<<1, 1>>>(v, d_result);
    hipMemcpy(&h_result, d_result, sizeof(float), hipMemcpyDeviceToHost);

    printf("norm(3,4) = %f\n", h_result); // Esperado: 5.000000

    hipFree(d_result);
    return 0;
}
