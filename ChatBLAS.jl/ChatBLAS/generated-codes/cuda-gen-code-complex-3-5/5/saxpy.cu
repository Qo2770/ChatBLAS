#include "chatblas_cuda.h"

__global__ void saxpy_kernel(int n, float a, float *x, float *y) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n) {
    y[tid] = a * x[tid] + y[tid];
  }
}

void chatblas_saxpy(int n, float a, float *x, float *y) {
  float *device_x, *device_y;
  cudaMalloc((void**)&device_x, sizeof(float) * n);
  cudaMalloc((void**)&device_y, sizeof(float) * n);

  cudaMemcpy(device_x, x, sizeof(float) * n, cudaMemcpyHostToDevice);
  cudaMemcpy(device_y, y, sizeof(float) * n, cudaMemcpyHostToDevice);

  int blockSize = 256;
  int numBlocks = (n + blockSize - 1) / blockSize;

  saxpy_kernel<<<numBlocks, blockSize>>>(n, a, device_x, device_y);

  cudaMemcpy(y, device_y, sizeof(float) * n, cudaMemcpyDeviceToHost);

  cudaFree(device_x);
  cudaFree(device_y);
}
