#include "chatblas_cuda.h"

__global__
void sswap(int n, float *x, float *y) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        float temp = x[idx];
        x[idx] = y[idx];
        y[idx] = temp;
    }
}

void chatblas_sswap(int n, float *x, float *y) {
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    sswap<<<gridSize, blockSize>>>(n, x, y);
    cudaDeviceSynchronize();
}
