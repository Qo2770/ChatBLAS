#include "chatblas_cuda.h"

__global__ void sdot_kernel(int n, float *x, float *y, float *res) {
  __shared__ float partialSum[256];

  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  float sum = 0;

  for (int i = tid; i < n; i += stride) {
    sum += x[i] * y[i];
  }

  partialSum[threadIdx.x] = sum;

  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      partialSum[threadIdx.x] += partialSum[threadIdx.x + stride];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    atomicAdd(res, partialSum[0]);
  }
}

float chatblas_sdot(int n, float *x, float *y) {
  float *dev_x, *dev_y, *dev_res;
  float res = 0.0;

  cudaMalloc((void**)&dev_x, n * sizeof(float));
  cudaMalloc((void**)&dev_y, n * sizeof(float));
  cudaMalloc((void**)&dev_res, sizeof(float));

  cudaMemcpy(dev_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_y, y, n * sizeof(float), cudaMemcpyHostToDevice);

  int blockSize = 256;
  int numBlocks = (n + blockSize - 1) / blockSize;

  sdot_kernel<<<numBlocks, blockSize>>>(n, dev_x, dev_y, dev_res);

  cudaMemcpy(&res, dev_res, sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(dev_x);
  cudaFree(dev_y);
  cudaFree(dev_res);

  return res;
}

