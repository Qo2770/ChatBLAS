#include "chatblas_cuda.h"

__global__ void copyKernel(int n, float *x, float *y) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        y[tid] = x[tid];
    }
}

void chatblas_scopy(int n, float *x, float *y) {
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    copyKernel<<<gridSize, blockSize>>>(n, x, y);

    cudaDeviceSynchronize();
}