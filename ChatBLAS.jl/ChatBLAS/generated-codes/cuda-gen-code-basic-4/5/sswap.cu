#include "chatblas_cuda.h"

// Kernel to swap values of two vectors
__global__ void swapKernel(int n, float *x, float *y) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    if (index < n) {
        float temp = x[index];
        x[index] = y[index];
        y[index] = temp;
    }
}

void chatblas_sswap(int n, float *x, float *y) {
    // Calculate optimal block size
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    // Get device pointers
    float *dev_x, *dev_y;
    cudaMalloc((void**)&dev_x, n * sizeof(float));
    cudaMalloc((void**)&dev_y, n * sizeof(float));

    // Copy inputs to device
    cudaMemcpy(dev_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_y, y, n * sizeof(float), cudaMemcpyHostToDevice);

    // Launch swapKernel() with numBlocks blocks each with blockSize threads
    swapKernel<<<numBlocks, blockSize>>>(n, dev_x, dev_y);

    // Copy output vector from device memory to host memory
    cudaMemcpy(x, dev_x, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(y, dev_y, n * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(dev_x);
    cudaFree(dev_y);
}
