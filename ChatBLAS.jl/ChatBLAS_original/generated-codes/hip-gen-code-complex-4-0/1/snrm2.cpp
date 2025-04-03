#include "chatblas_hip.h"

__global__ void snrm2_kernel(int n, float *x, float *res) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    float sum = 0.0;

    if (idx < n) {
        sum += x[idx] * x[idx];
    }

    atomicAdd(res, sum);
}

float chatblas_snrm2(int n, float *x) {
    float *x_gpu, *res_gpu;
    float res;
    const int blocks = (n + 255) / 256;

    hipMalloc((void**)&x_gpu, n * sizeof(float));
    hipMalloc((void**)&res_gpu, sizeof(float));
    hipMemcpy(x_gpu, x, n * sizeof(float), hipMemcpyHostToDevice);

    snrm2_kernel<<<blocks, 256>>>(n, x_gpu, res_gpu);

    hipMemcpy(&res, res_gpu, sizeof(float), hipMemcpyDeviceToHost);

    hipFree(x_gpu);
    hipFree(res_gpu);

    return sqrt(res);
}