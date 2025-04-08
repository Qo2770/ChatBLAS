#include "chatblas_cuda.h"
#include <cuda_runtime.h>
#include <stdio.h>

// CUDA kernel for copying elements from x to y
__global__ void copyKernel(int n, float *x, float *y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = x[i];
    }
}

// Function to copy a vector x to a vector y in parallel using CUDA
void chatblas_scopy(int n, float *x, float *y) {
    float *d_x, *d_y;
    size_t size = n * sizeof(float);

    // Allocate device memory
    cudaMalloc((void **)&d_x, size);
    cudaMalloc((void **)&d_y, size);

    // Copy data from host to device
    cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice);

    // Determine block and grid sizes
    int blockSize = 256; // You can choose different values based on your hardware
    int gridSize = (n + blockSize - 1) / blockSize;

    // Launch the kernel to copy data from x to y
    copyKernel<<<gridSize, blockSize>>>(n, d_x, d_y);

    // Copy the result from device to host
    cudaMemcpy(y, d_y, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_x);
    cudaFree(d_y);
}
