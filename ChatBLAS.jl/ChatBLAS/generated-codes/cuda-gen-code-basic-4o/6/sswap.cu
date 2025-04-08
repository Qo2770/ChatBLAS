#include "chatblas_cuda.h"

__global__ void swap_kernel(int n, float *x, float *y) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        float temp = x[index];
        x[index] = y[index];
        y[index] = temp;
    }
}

void chatblas_sswap(int n, float *x, float *y) {
    // Assuming x_d and y_d are device pointers, allocate and copy to device
    float *x_d, *y_d;
    cudaMalloc((void**)&x_d, n * sizeof(float));
    cudaMalloc((void**)&y_d, n * sizeof(float));
    
    cudaMemcpy(x_d, x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(y_d, y, n * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    swap_kernel<<<numBlocks, blockSize>>>(n, x_d, y_d);

    // Copy device arrays back to host arrays
    cudaMemcpy(x, x_d, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(y, y_d, n * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(x_d);
    cudaFree(y_d);
}
