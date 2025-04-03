#include "chatblas_cuda.h"

__global__ void sswap_kernel(int n, float *x, float *y) {
  int index = threadIdx.x + blockDim.x * blockIdx.x;
  float temp;

  if (index < n) {
    temp = x[index];
    x[index] = y[index];
    y[index] = temp;
  }
}

void chatblas_sswap(int n, float *x, float *y) {
  float *device_x, *device_y;

  cudaMalloc(&device_x, n*sizeof(float));
  cudaMalloc(&device_y, n*sizeof(float));

  cudaMemcpy(device_x, x, n*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(device_y, y, n*sizeof(float), cudaMemcpyHostToDevice);

  int blockSize = 256;
  int numBlocks = (n + blockSize - 1) / blockSize;
  sswap_kernel<<<numBlocks, blockSize>>>(n, device_x, device_y);

  cudaMemcpy(x, device_x, n*sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(y, device_y, n*sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(device_x);
  cudaFree(device_y);
}