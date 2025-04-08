#include "chatblas_cuda.h"

__global__ void saxpyKernel(int n, float a, float *x, float *y) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        y[index] = a * x[index] + y[index];
    }
}

void chatblas_saxpy(int n, float a, float *x, float *y) {
    float *d_x, *d_y;
    size_t size = n * sizeof(float);
    
    cudaMalloc(&d_x, size);
    cudaMalloc(&d_y, size);
    
    cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, size, cudaMemcpyHostToDevice);
    
    int blockSize = 256; // Number of threads in each block
    int numBlocks = (n + blockSize - 1) / blockSize; // Number of blocks
    
    saxpyKernel<<<numBlocks, blockSize>>>(n, a, d_x, d_y);
    
    cudaMemcpy(y, d_y, size, cudaMemcpyDeviceToHost);
    
    cudaFree(d_x);
    cudaFree(d_y);
}
