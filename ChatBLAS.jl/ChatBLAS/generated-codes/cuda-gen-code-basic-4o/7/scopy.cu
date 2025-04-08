#include "chatblas_cuda.h"
#include <cuda_runtime.h>
#include <stdio.h>

// CUDA kernel that performs the copy operation
__global__ void vectorCopyKernel(int n, const float *x, float *y) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = x[idx];
    }
}

// Function that copies vector x to vector y
void chatblas_scopy(int n, float *x, float *y) {
    float *d_x, *d_y;
    size_t size = n * sizeof(float);

    // Allocate device memory
    cudaMalloc((void **)&d_x, size);
    cudaMalloc((void **)&d_y, size);

    // Copy x from host to device
    cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice);

    // Calculate block and grid sizes
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the kernel
    vectorCopyKernel<<<blocksPerGrid, threadsPerBlock>>>(n, d_x, d_y);

    // Wait for the GPU to finish
    cudaDeviceSynchronize();

    // Copy the result from device to host
    cudaMemcpy(y, d_y, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_x);
    cudaFree(d_y);
}
