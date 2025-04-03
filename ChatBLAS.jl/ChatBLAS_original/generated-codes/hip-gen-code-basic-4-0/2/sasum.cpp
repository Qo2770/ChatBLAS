#include "hip/hip_runtime.h"

__global__ void sasum_kernel(int n, float *x, float *sum) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    atomicAdd(sum, fabs(x[i]));
  }
}

float chatblas_sasum(int n, float *x) {

  float *d_x, *d_sum;
  float sum = 0.0f;

  hipMalloc((void **)&d_x, n * sizeof(float));
  hipMalloc((void **)&d_sum, sizeof(float));

  hipMemcpy(d_x, x, n * sizeof(float), hipMemcpyHostToDevice);
  hipMemcpy(d_sum, &sum, sizeof(float), hipMemcpyHostToDevice);
  
  int numThreads = 256;
  int numBlocks = (n + numThreads - 1) / numThreads;

  hipLaunchKernelGGL(sasum_kernel, numBlocks, numThreads, 0, 0, n, d_x, d_sum);

  hipMemcpy(&sum, d_sum, sizeof(float), hipMemcpyDeviceToHost);

  hipFree(d_x);
  hipFree(d_sum);

  return sum;
}
