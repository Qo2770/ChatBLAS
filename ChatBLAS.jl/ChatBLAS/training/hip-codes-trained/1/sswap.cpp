#include "chatblas_hip.h" 

__global__ void sswap_kernel(int n, float *x, float *y) { int index = threadIdx.x + blockIdx.x * blockDim.x; if (index < n) { float temp = x[index]; x[index] = y[index]; y[index] = temp; } } 

void chatblas_sswap(int n, float *x, float *y) { float *d_x, *d_y; int blockSize = 256; int numBlocks = (n + blockSize - 1) / blockSize; hipMalloc((void**)&d_x, n * sizeof(float)); hipMalloc((void**)&d_y, n * sizeof(float)); hipMemcpy(d_x, x, n * sizeof(float), hipMemcpyHostToDevice); hipMemcpy(d_y, y, n * sizeof(float), hipMemcpyHostToDevice); sswap_kernel<<<numBlocks, blockSize>>>(n, d_x, d_y); hipMemcpy(x, d_x, n * sizeof(float), hipMemcpyDeviceToHost); hipMemcpy(y, d_y, n * sizeof(float), hipMemcpyDeviceToHost); hipFree(d_x); hipFree(d_y); }
