#include "chatblas_hip.h"

__global__ void saxpy_kernel(int n, float a, float *x, float *y) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) y[i] = a * x[i] + y[i];
}

void chatblas_saxpy(int n, float a, float *x, float *y) {

  float *x_dev = NULL, *y_dev = NULL;
  hipMalloc((void**)&x_dev, n * sizeof(float));
  hipMalloc((void**)&y_dev, n * sizeof(float));

  hipMemcpy(x_dev, x, n * sizeof(float), hipMemcpyHostToDevice);
  hipMemcpy(y_dev, y, n * sizeof(float), hipMemcpyHostToDevice);

  int blockSize = 256;
  int gridSize = (n + blockSize - 1) / blockSize;
  hipLaunchKernelGGL(saxpy_kernel, dim3(gridSize), dim3(blockSize), 0, 0, n, a, x_dev, y_dev);

  hipMemcpy(y, y_dev, n * sizeof(float), hipMemcpyDeviceToHost);

  hipFree(x_dev);
  hipFree(y_dev);
}
