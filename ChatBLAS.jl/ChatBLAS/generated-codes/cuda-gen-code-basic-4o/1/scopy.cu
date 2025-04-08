#include "chatblas_cuda.h"
#include <cuda_runtime.h>
#include <stdio.h>

// CUDA kernel to copy elements from vector x to vector y
__global__ void scopy_kernel(int n, const float *x, float *y) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = x[idx];
    }
}

// Function to copy vector x to vector y using CUDA
void chatblas_scopy(int n, float *x, float *y) {
    // Allocate device memory
    float *d_x, *d_y;
    cudaMalloc((void**)&d_x, n * sizeof(float));
    cudaMalloc((void**)&d_y, n * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);

    // Define block size and compute number of blocks
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    // Launch the CUDA kernel
    scopy_kernel<<<numBlocks, blockSize>>>(n, d_x, d_y);

    // Copy the result from device to host
    cudaMemcpy(y, d_y, n * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_x);
    cudaFree(d_y);
}
