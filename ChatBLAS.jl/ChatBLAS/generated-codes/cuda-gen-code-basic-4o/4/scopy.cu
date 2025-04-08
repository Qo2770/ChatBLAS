#include "chatblas_cuda.h"
#include "chatblas_cuda.h"
#include <cuda_runtime.h>
#include <stdio.h>

// CUDA kernel to copy elements from vector x to vector y
__global__ void scopy_kernel(int n, float *x, float *y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < n) {
        y[i] = x[i];
    }
}

// Host function to copy vector x to vector y
void chatblas_scopy(int n, float *x, float *y) {
    float *d_x, *d_y;

    // Allocate device memory for vectors
    cudaMalloc((void**)&d_x, n * sizeof(float));
    cudaMalloc((void**)&d_y, n * sizeof(float));

    // Copy the vector x from host to device
    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);

    // Define grid and block size
    int blockSize = 256; // you can adjust this according to your needs and device properties
    int gridSize = (n + blockSize - 1) / blockSize;

    // Launch the kernel to copy vector x to vector y
    scopy_kernel<<<gridSize, blockSize>>>(n, d_x, d_y);

    // Copy the result vector y from device to host
    cudaMemcpy(y, d_y, n * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_x);
    cudaFree(d_y);
}
