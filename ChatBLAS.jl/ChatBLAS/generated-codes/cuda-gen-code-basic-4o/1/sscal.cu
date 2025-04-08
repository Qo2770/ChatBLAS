#include "chatblas_cuda.h"
#include <cuda_runtime.h>
#include <stdio.h>

// CUDA Kernel function for scaling the vector
__global__ void sscal_kernel(int n, float a, float *x) {
    // Calculate the global index for each thread
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    
    // Only process if the index is within bounds
    if (idx < n) {
        x[idx] *= a;
    }
}

// Function to scale the vector using CUDA
void chatblas_sscal(int n, float a, float *x) {
    // Define pointers for device
    float *d_x;
    
    // Allocate memory on the device
    cudaMalloc(&d_x, n * sizeof(float));
    
    // Copy data from host to device
    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
    
    // Define block and grid dimensions
    int blockSize = 256; // Number of threads per block
    int gridSize = (n + blockSize - 1) / blockSize; // Number of blocks
    
    // Launch the kernel
    sscal_kernel<<<gridSize, blockSize>>>(n, a, d_x);
    
    // Check for any errors launching the kernel
    cudaError_t cudaErr = cudaGetLastError();
    if (cudaErr != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(cudaErr));
        return;
    }
    
    // Copy the result from device to host
    cudaMemcpy(x, d_x, n * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(d_x);
}
