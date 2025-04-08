// chatblas_cuda.h
#ifndef CHATBLAS_CUDA_H
#define CHATBLAS_CUDA_H

#include <cuda_runtime.h>

// Add any additional declarations or helper functions that might be needed

#endif // CHATBLAS_CUDA_H
#include "chatblas_cuda.h"

// CUDA kernel to perform vector copy
__global__ void vectorCopyKernel(int n, const float *x, float *y) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        y[index] = x[index];
    }
}

void chatblas_scopy(int n, float *x, float *y) {
    // Allocate GPU memory for input and output vectors
    float *d_x, *d_y;
    size_t size = n * sizeof(float);
    
    cudaMalloc((void**)&d_x, size);
    cudaMalloc((void**)&d_y, size);
    
    // Copy input vector from host to device
    cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice);
    
    // Define block and grid sizes
    int blockSize = 256; // You can set this to an optimal size based on your GPU architecture
    int gridSize = (n + blockSize - 1) / blockSize;
    
    // Launch kernel to perform vector copy on the GPU
    vectorCopyKernel<<<gridSize, blockSize>>>(n, d_x, d_y);
    
    // Copy result from device back to host
    cudaMemcpy(y, d_y, size, cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(d_x);
    cudaFree(d_y);
}
