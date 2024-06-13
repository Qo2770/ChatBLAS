#include "chatblas_cuda.h"

__global__ void swapKernel(int n, float *x, float *y) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        float temp = x[index];
        x[index] = y[index];
        y[index] = temp;
    }
}

void chatblas_sswap(int n, float *x, float *y) {
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    swapKernel<<<gridSize, blockSize>>>(n, x, y);
    cudaDeviceSynchronize();
}
