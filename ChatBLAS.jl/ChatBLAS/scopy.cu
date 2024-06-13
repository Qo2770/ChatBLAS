#include "chatblas_cuda.h" 

__global__ void scopy_kernel(int n, float *x, float *y) { int index = blockIdx.x * blockDim.x + threadIdx.x; int stride = blockDim.x * gridDim.x; for (int i = index; i < n; i += stride) { y[i] = x[i]; } } 

void chatblas_scopy(int n, float *x, float *y) { float *d_x, *d_y; int size = n * sizeof(float); cudaMalloc((void **)&d_x, size); cudaMalloc((void **)&d_y, size); cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice); int blockSize = 256; int numBlocks = (n + blockSize - 1) / blockSize; scopy_kernel<<<numBlocks, blockSize>>>(n, d_x, d_y); cudaMemcpy(y, d_y, size, cudaMemcpyDeviceToHost); cudaFree(d_x); cudaFree(d_y); }
