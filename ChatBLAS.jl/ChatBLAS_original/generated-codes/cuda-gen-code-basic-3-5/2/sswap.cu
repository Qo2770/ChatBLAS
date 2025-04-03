#include "chatblas_cuda.h"

__global__ void swapKernel(int n, float *x, float *y) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        float temp = x[index];
        x[index] = y[index];
        y[index] = temp;
    }
}

void chatblas_sswap(int n, float *x, float *y) {
    // Allocate device memory
    float *dev_x, *dev_y;
    cudaMalloc(&dev_x, n * sizeof(float));
    cudaMalloc(&dev_y, n * sizeof(float));
    
    // Copy input vectors from host to device memory
    cudaMemcpy(dev_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_y, y, n * sizeof(float), cudaMemcpyHostToDevice);
    
    // Perform swap on device
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    swapKernel<<<gridSize, blockSize>>>(n, dev_x, dev_y);
    
    // Copy results back from device to host memory
    cudaMemcpy(x, dev_x, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(y, dev_y, n * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(dev_x);
    cudaFree(dev_y);
}
