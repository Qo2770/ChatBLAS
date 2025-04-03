#include "chatblas_cuda.h"
#include <cuda.h>

__global__ void axpyKernel(int n, float a, float *x, float *y) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        y[i] = a * x[i] + y[i];
    }
}

void chatblas_saxpy(int n, float a, float *x, float *y) {
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    
    float *d_x, *d_y;
    cudaMalloc(&d_x, n * sizeof(float));
    cudaMalloc(&d_y, n * sizeof(float));

    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, n * sizeof(float), cudaMemcpyHostToDevice);

    axpyKernel<<<numBlocks, blockSize>>>(n, a, d_x, d_y);

    cudaMemcpy(y, d_y, n * sizeof(float), cudaMemcpyDeviceToHost);
  
    cudaFree(d_x);
    cudaFree(d_y);
}
