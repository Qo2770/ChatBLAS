#include "chatblas_cuda.h"

__global__ void sdot_kernel(int n, float *x, float *y, float *res) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    float result = 0;
    if (index < n) {
        result = x[index] * y[index];
    }

    atomicAdd(res, result);
}

float chatblas_sdot( int n, float *x, float *y) {
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    float *d_x, *d_y, *d_res;
    float res = 0;

    cudaMalloc(&d_x, n*sizeof(float));
    cudaMemcpy(d_x, x, n*sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&d_y, n*sizeof(float));
    cudaMemcpy(d_y, y, n*sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&d_res, sizeof(float));
    cudaMemcpy(d_res, &res, sizeof(float), cudaMemcpyHostToDevice);

    sdot_kernel<<<gridSize, blockSize>>>(n, d_x, d_y, d_res);

    cudaMemcpy(&res, d_res, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_res);

    return res;
}