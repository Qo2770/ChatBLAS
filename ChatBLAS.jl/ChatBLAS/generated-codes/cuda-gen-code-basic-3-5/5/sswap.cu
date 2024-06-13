#include "chatblas_cuda.h"

__global__ void swapKernel(int n, float *x, float *y) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float tmp = x[idx];
        x[idx] = y[idx];
        y[idx] = tmp;
    }
}

void chatblas_sswap(int n, float *x, float *y) {
    int numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    swapKernel<<<numBlocks, BLOCK_SIZE>>>(n, x, y);
    cudaDeviceSynchronize();
}
