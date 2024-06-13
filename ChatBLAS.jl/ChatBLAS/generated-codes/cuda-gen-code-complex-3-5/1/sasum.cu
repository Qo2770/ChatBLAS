#include "chatblas_cuda.h"

__global__ void sasum_kernel(int n, float *x, float *sum) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  float partialSum = 0.0f;
  for (int i = tid; i < n; i += stride) {
    partialSum += abs(x[i]);
  }

  atomicAdd(sum, partialSum);
}

float chatblas_sasum(int n, float *x) {
  float *dev_x, *dev_sum;
  float sum = 0.0f;

  cudaMalloc((void**)&dev_x, n * sizeof(float));
  cudaMalloc((void**)&dev_sum, sizeof(float));

  cudaMemcpy(dev_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_sum, &sum, sizeof(float), cudaMemcpyHostToDevice);

  int blockSize = 256;
  int numBlocks = (n + blockSize - 1) / blockSize;

  sasum_kernel<<<numBlocks, blockSize>>>(n, dev_x, dev_sum);

  cudaMemcpy(&sum, dev_sum, sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(dev_x);
  cudaFree(dev_sum);

  return sum;
}