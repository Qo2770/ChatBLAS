#include "chatblas_hip.h"
__global__ void snrm2_kernel(int n, float *x, float *res) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;

    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        sum += x[i] * x[i];
    }

    atomicAdd(res, sum);
}

float chatblas_snrm2(int n, float *x) {
    float *d_x, *d_res;
    float h_res = 0.0f;

    hipMalloc(&d_x, n * sizeof(float));
    hipMalloc(&d_res, sizeof(float));

    hipMemcpy(d_x, x, n * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_res, &h_res, sizeof(float), hipMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    snrm2_kernel<<<numBlocks, blockSize>>>(n, d_x, d_res);

    hipMemcpy(&h_res, d_res, sizeof(float), hipMemcpyDeviceToHost);

    hipFree(d_x);
    hipFree(d_res);

    return sqrt(h_res);
}