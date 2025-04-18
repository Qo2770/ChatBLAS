#include "chatblas_cuda.h"

__global__ void swapKernel(int n, float *x, float *y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float temp = x[i];
        x[i] = y[i];
        y[i] = temp;
    }
}

void chatblas_sswap(int n, float *x, float *y) {
    float *d_x, *d_y;
    size_t size = n * sizeof(float);

    // Allocate memory on the device
    cudaMalloc((void **)&d_x, size);
    cudaMalloc((void **)&d_y, size);

    // Copy vectors from host to device
    cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, size, cudaMemcpyHostToDevice);

    // Define block and grid sizes
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    // Launch the kernel
    swapKernel<<<numBlocks, blockSize>>>(n, d_x, d_y);

    // Copy the swapped vectors back to the host
    cudaMemcpy(x, d_x, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(y, d_y, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_x);
    cudaFree(d_y);
}
