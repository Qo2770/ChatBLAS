#include "chatblas_cuda.h"

__global__ void sscal_kernel(int n, float a, float *x) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < n) {
    x[tid] *= a;
  }
}

void chatblas_sscal(int n, float a, float *x) {
  float *d_x;

  // Allocate GPU memory
  cudaMalloc((void **)&d_x, n * sizeof(float));

  // Copy input vector from host to GPU
  cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);

  // Calculate block size and number of blocks
  int blockSize = 256;
  int numBlocks = (n + blockSize - 1) / blockSize;

  // Launch kernel
  sscal_kernel<<<numBlocks, blockSize>>>(n, a, d_x);

  // Copy output vector from GPU to host
  cudaMemcpy(x, d_x, n * sizeof(float), cudaMemcpyDeviceToHost);

  // Free GPU memory
  cudaFree(d_x);
}
