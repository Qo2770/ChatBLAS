#include "chatblas_cuda.h"

__global__ void sdsdot_kernel(int n, float b, float *x, float *y, double *res) {
  int index = threadIdx.x + blockDim.x * blockIdx.x;
  double result = 0.0;

  if (index < n) {
    result = ((double) x[index]) * ((double) y[index]);
    atomicAdd(res, result);
  }

  if (index == 0) {
    *res += (double)b;
  }
}

float chatblas_sdsdot(int n, float b, float *x, float *y) {
  float *dev_x = 0;
  float *dev_y = 0;
  double *dev_res = 0;
  double h_res = 0.0;

  cudaMalloc((void**)&dev_x, n * sizeof(float));
  cudaMalloc((void**)&dev_y, n * sizeof(float));
  cudaMalloc((void**)&dev_res, sizeof(double));

  cudaMemcpy(dev_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_y, y, n * sizeof(float), cudaMemcpyHostToDevice);

  int blockSize = 256;
  int numBlocks = (n + blockSize - 1) / blockSize;

  sdsdot_kernel<<<numBlocks, blockSize>>>(n, b, dev_x, dev_y, dev_res);

  cudaMemcpy(&h_res, dev_res, sizeof(double), cudaMemcpyDeviceToHost);

  cudaFree(dev_x);
  cudaFree(dev_y);
  cudaFree(dev_res);

  return (float)h_res;
}
