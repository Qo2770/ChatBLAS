#include "chatblas_hip.h"

__global__ void snrm2_kernel(int n, float *x, float *res) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        atomicAdd(res, x[idx] * x[idx]);
    }
}

float chatblas_snrm2(int n, float *x) {
    float *d_x, *d_res;
    float res;
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    hipMalloc((void**)&d_x, n * sizeof(float));
    hipMalloc((void**)&d_res, sizeof(float));
    hipMemcpy(d_x, x, n * sizeof(float), hipMemcpyHostToDevice);

    snrm2_kernel<<<gridSize, blockSize>>>(n, d_x, d_res);

    hipMemcpy(&res, d_res, sizeof(float), hipMemcpyDeviceToHost);

    res = sqrt(res);

    hipFree(d_x);
    hipFree(d_res);

    return res;
}