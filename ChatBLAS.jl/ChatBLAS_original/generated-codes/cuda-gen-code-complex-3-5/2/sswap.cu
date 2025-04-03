#include "chatblas_cuda.h"

__global__ void sswap_kernel(int n, float *x, float *y) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < n) {
    float temp = x[i];
    x[i] = y[i];
    y[i] = temp;
  }
}

void chatblas_sswap(int n, float *x, float *y) {
  float *d_x, *d_y;

  // Allocate GPU memory
  cudaMalloc(&d_x, n * sizeof(float));
  cudaMalloc(&d_y, n * sizeof(float));

  // Transfer data from CPU to GPU
  cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, n * sizeof(float), cudaMemcpyHostToDevice);

  // Define the number of blocks and threads per block
  int threadsPerBlock = 256;
  int numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;

  // Call the kernel function
  sswap_kernel<<<numBlocks, threadsPerBlock>>>(n, d_x, d_y);

  // Transfer data from GPU to CPU
  cudaMemcpy(x, d_x, n * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(y, d_y, n * sizeof(float), cudaMemcpyDeviceToHost);

  // Free GPU memory
  cudaFree(d_x);
  cudaFree(d_y);
}
