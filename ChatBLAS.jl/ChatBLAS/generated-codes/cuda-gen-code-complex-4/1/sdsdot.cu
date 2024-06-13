#include "chatblas_cuda.h"

__global__ void sdsdot_kernel(int n, float b, float *x, float *y, float *res) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n) {
        res[tid] = static_cast<double>(x[tid]) * static_cast<double>(y[tid]);
    }
}

float chatblas_sdsdot(int n, float b, float *x, float *y) {
    float *d_x, *d_y, *d_res;
    float result = 0.0;

    cudaMalloc((void**)&d_x, n * sizeof(float));
    cudaMalloc((void**)&d_y, n * sizeof(float));
    cudaMalloc((void**)&d_res, n * sizeof(float));

    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, n * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    sdsdot_kernel<<<numBlocks, blockSize>>>(n, b, d_x, d_y, d_res);

    cudaMemcpy(&result, d_res, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_res);

    return result + b;
}

