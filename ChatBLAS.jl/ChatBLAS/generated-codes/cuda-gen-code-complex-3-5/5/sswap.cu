#include "chatblas_cuda.h"

__global__ void sswap_kernel(int n, float *x, float *y) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n) {
    float temp = x[tid];
    x[tid] = y[tid];
    y[tid] = temp;
  }
}

void chatblas_sswap(int n, float *x, float *y) {
  int numThreadsPerBlock = 256;
  int numBlocks = (n + numThreadsPerBlock - 1) / numThreadsPerBlock;
  
  float *d_x, *d_y;
  cudaMalloc((void **)&d_x, n * sizeof(float));
  cudaMalloc((void **)&d_y, n * sizeof(float));
  
  cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, n * sizeof(float), cudaMemcpyHostToDevice);
  
  sswap_kernel<<<numBlocks, numThreadsPerBlock>>>(n, d_x, d_y);
  
  cudaMemcpy(x, d_x, n * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(y, d_y, n * sizeof(float), cudaMemcpyDeviceToHost);
  
  cudaFree(d_x);
  cudaFree(d_y);
}
