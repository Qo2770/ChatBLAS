#include "chatblas_cuda.h" 

__global__ void sswap_kernel(int n, float *x, float *y) { int index = blockIdx.x * blockDim.x + threadIdx.x; int stride = blockDim.x * gridDim.x; for (int i = index; i < n; i += stride) { float temp = x[i]; x[i] = y[i]; y[i] = temp; } } 

void chatblas_sswap(int n, float *x, float *y) { float *d_x, *d_y; int blockSize = 256; int numBlocks = (n + blockSize - 1) / blockSize; cudaMalloc((void **)&d_x, n * sizeof(float)); cudaMalloc((void **)&d_y, n * sizeof(float)); cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice); cudaMemcpy(d_y, y, n * sizeof(float), cudaMemcpyHostToDevice); sswap_kernel<<<numBlocks, blockSize>>>(n, d_x, d_y); cudaMemcpy(x, d_x, n * sizeof(float), cudaMemcpyDeviceToHost); cudaMemcpy(y, d_y, n * sizeof(float), cudaMemcpyDeviceToHost); cudaFree(d_x); cudaFree(d_y); }
