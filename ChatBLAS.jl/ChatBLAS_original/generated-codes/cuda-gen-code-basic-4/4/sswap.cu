#include "chatblas_cuda.h"

__global__ void swapKernel(int n, float *x, float *y) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float temp = x[i];
    x[i] = y[i];
    y[i] = temp;
  }
}

void chatblas_sswap(int n, float *x, float *y) {
  int blockSize = 256;
  int numBlocks = (n + blockSize - 1) / blockSize;

  swapKernel<<<numBlocks, blockSize>>>(n, x, y);

  cudaDeviceSynchronize();
}
