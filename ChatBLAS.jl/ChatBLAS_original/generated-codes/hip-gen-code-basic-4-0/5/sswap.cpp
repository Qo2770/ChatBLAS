#include "chatblas_hip.h"

__global__ void sswap_kernel(int n, float *x, float *y) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i < n) {
    float t = x[i];
    x[i] = y[i];
    y[i] = t;
  }
}

void chatblas_sswap(int n, float *x, float *y) {
  float *dx, *dy;
  hipMalloc((void**)&dx, n * sizeof(float));
  hipMalloc((void**)&dy, n * sizeof(float));

  hipMemcpy(dx, x, n * sizeof(float), hipMemcpyHostToDevice);
  hipMemcpy(dy, y, n * sizeof(float), hipMemcpyHostToDevice);

  dim3 blockSize(256);
  dim3 gridSize((n + blockSize.x - 1) / blockSize.x);
  hipLaunchKernelGGL(sswap_kernel, gridSize, blockSize, 0, 0, n, dx, dy);

  hipMemcpy(x, dx, n * sizeof(float), hipMemcpyDeviceToHost);
  hipMemcpy(y, dy, n * sizeof(float), hipMemcpyDeviceToHost);

  hipFree(dx);
  hipFree(dy);
}
