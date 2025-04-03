#include "chatblas_cuda.h"

__global__ void saxpy_kernel(int n, float a, float *x, float *y) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = a * x[i] + y[i];
}

void chatblas_saxpy(int n, float a, float *x, float *y) {
  int size = n * sizeof(float);
  float *d_x, *d_y;

  cudaMalloc(&d_x, size);
  cudaMalloc(&d_y, size);

  cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, size, cudaMemcpyHostToDevice);

  saxpy_kernel<<<(n+255)/256, 256>>>(n, a, d_x, d_y);

  cudaMemcpy(y, d_y, size, cudaMemcpyDeviceToHost);

  cudaFree(d_x);
  cudaFree(d_y);
}