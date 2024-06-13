#include "chatblas_cuda.h" 
__global__ void isamax_kernel(int n, float *x, float *ind) { int i = blockDim.x * blockIdx.x + threadIdx.x; if (i < n) { if (abs(x[i]) > abs(x[(int)(*ind)])) { *ind = (float)i; } } } 

int chatblas_isamax(int n, float *x) { float *x_d, *ind_d; float ind_h = 0; cudaMalloc((void **)&x_d, n * sizeof(float)); cudaMalloc((void **)&ind_d, sizeof(float)); cudaMemcpy(x_d, x, n * sizeof(float), cudaMemcpyHostToDevice); cudaMemcpy(ind_d, &ind_h, sizeof(float), cudaMemcpyHostToDevice); int blockSize = 256; int numBlocks = (n + blockSize - 1) / blockSize; isamax_kernel<<<numBlocks, blockSize>>>(n, x_d, ind_d); cudaMemcpy(&ind_h, ind_d, sizeof(float), cudaMemcpyDeviceToHost); cudaFree(x_d); cudaFree(ind_d); return (int) ind_h; }
