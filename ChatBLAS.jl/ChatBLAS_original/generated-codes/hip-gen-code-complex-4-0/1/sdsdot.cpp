#include "chatblas_hip.h"

__global__ void sdsdot_kernel( int n, float b, float *x, float *y, float *res ) {
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    __shared__ double tmp[WARP_SIZE];
    tmp[threadIdx.x] = 0;

    if (idx >= n) return;
    tmp[threadIdx.x] = (double)x[idx] * (double)y[idx];

    __syncthreads();

    if (threadIdx.x == 0) {
        double sum = 0;
        for (int i = 0; i < WARP_SIZE; ++i) {
            sum += tmp[i];
        }
        atomicAddD(res, sum);
    }
}

float chatblas_sdsdot( int n, float b, float *x, float *y) {
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    float* d_x;
    float* d_y;
    float* d_res;
    float result = 0;

    hipMalloc((void**)&d_x, n * sizeof(float));
    hipMalloc((void**)&d_y, n * sizeof(float));
    hipMalloc((void**)&d_res, sizeof(float));

    hipMemcpy(d_x, x, n * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_y, y, n * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_res, &result, sizeof(float), hipMemcpyHostToDevice);

    sdsdot_kernel<<<gridSize, blockSize>>>(n, b, d_x, d_y, d_res);

    hipMemcpy(&result, d_res, sizeof(float), hipMemcpyDeviceToHost);

    hipFree(d_x);
    hipFree(d_y);
    hipFree(d_res);

    return result + b;
}