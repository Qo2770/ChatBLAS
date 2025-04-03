#include "chatblas_hip.h" 

__global__ void saxpy_kernel(int n, float a, float *x, float *y) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) { y[i] = a * x[i] + y[i]; } } 

void chatblas_saxpy(int n, float a, float *x, float *y) { float *d_x, *d_y; hipMalloc((void**)&d_x, n * sizeof(float)); hipMalloc((void**)&d_y, n * sizeof(float)); hipMemcpy(d_x, x, n * sizeof(float), hipMemcpyHostToDevice); hipMemcpy(d_y, y, n * sizeof(float), hipMemcpyHostToDevice); int blockSize = 256; int numBlocks = (n + blockSize - 1) / blockSize; hipLaunchKernelGGL(saxpy_kernel, numBlocks, blockSize, 0, 0, n, a, d_x, d_y); hipMemcpy(y, d_y, n * sizeof(float), hipMemcpyDeviceToHost); hipFree(d_x); hipFree(d_y); }
