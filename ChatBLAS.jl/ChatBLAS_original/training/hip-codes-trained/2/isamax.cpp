#include "chatblas_hip.h" 

__global__ void isamax_kernel(int n, float *x, int *ind) { int i = blockIdx.x *blockDim.x+threadIdx.x; __shared__ int maxIndex; if(i == 0) { maxIndex = 0; } __syncthreads(); if(i < n) { if (abs(x[i]) > abs(x[maxIndex])) { maxIndex = i; } } __syncthreads(); if (i == 0) { *ind = maxIndex; } } 

int chatblas_isamax(int n, float *x) { float *d_x; int *ind, *d_ind; int blockSize = 32; int numBlocks = (n + blockSize - 1) / blockSize; ind = (int*)malloc(sizeof(int)); hipMalloc((void**)&d_x, n*sizeof(float)); hipMalloc((void**)&d_ind, sizeof(int)); hipMemcpy(d_x, x, n*sizeof(float), hipMemcpyHostToDevice); isamax_kernel<<<numBlocks, blockSize>>>(n, d_x, d_ind); hipMemcpy(ind, d_ind, sizeof(int), hipMemcpyDeviceToHost); int result = *ind; free(ind); hipFree(d_x); hipFree(d_ind); return result; }
