#include "chatblas_cuda.h"

__global__ void saxpy_kernel(int n, float a, float *x, float *y) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = a * x[i] + y[i];
}

void chatblas_saxpy(int n, float a, float *x, float *y) {
  float *d_x, *d_y;

  cudaMalloc((void **)&d_x, n * sizeof(float)); 
  cudaMalloc((void **)&d_y, n * sizeof(float));

  cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, n * sizeof(float), cudaMemcpyHostToDevice);

  saxpy_kernel<<<(n + 255) / 256, 256>>>(n, a, d_x, d_y);

  cudaMemcpy(y, d_y, n * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_x); 
  cudaFree(d_y);
}
