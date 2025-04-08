#include "chatblas_cuda.h"

// CUDA kernel to copy vector x to vector y
__global__ void copyKernel(int n, const float *x, float *y) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = x[idx];
    }
}

// Function to copy a vector x to vector y using CUDA
void chatblas_scopy(int n, float *x, float *y) {
    // Define CUDA grid and block sizes
    const int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    // Allocate device memory
    float *d_x, *d_y;
    cudaMalloc((void**)&d_x, n * sizeof(float));
    cudaMalloc((void**)&d_y, n * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);

    // Launch the kernel to perform the copy
    copyKernel<<<numBlocks, blockSize>>>(n, d_x, d_y);

    // Copy result from device to host
    cudaMemcpy(y, d_y, n * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_x);
    cudaFree(d_y);
}
