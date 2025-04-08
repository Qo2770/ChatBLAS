#include "chatblas_cuda.h"

// CUDA kernel to copy elements from vector x to vector y
__global__ void vectorCopyKernel(int n, float *x, float *y) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        y[idx] = x[idx];
    }
}

void chatblas_scopy(int n, float *x, float *y) {
    // Check if there is work to do
    if (n <= 0) return;

    // Define CUDA kernel launch parameters
    int threadBlockSize = 256; // common choice for block size
    int numBlocks = (n + threadBlockSize - 1) / threadBlockSize;

    // Allocate device memory pointers
    float *d_x, *d_y;
    
    // Allocate device memory
    cudaError_t err1 = cudaMalloc((void**)&d_x, n * sizeof(float));
    cudaError_t err2 = cudaMalloc((void**)&d_y, n * sizeof(float));
    
    if (err1 != cudaSuccess || err2 != cudaSuccess) {
        fprintf(stderr, "CUDA memory allocation failed\n");
        // Handle error here (e.g., exit, or return an error code)
        return;
    }

    // Copy data from host to device
    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);

    // Launch the kernel
    vectorCopyKernel<<<numBlocks, threadBlockSize>>>(n, d_x, d_y);

    // Ensure kernel launch was successful
    cudaError_t errSync = cudaGetLastError();
    cudaError_t errAsync = cudaDeviceSynchronize();
    if (errSync != cudaSuccess) fprintf(stderr, "Kernel launch error: %s\n", cudaGetErrorString(errSync));
    if (errAsync != cudaSuccess) fprintf(stderr, "Kernel execution error: %s\n", cudaGetErrorString(errAsync));

    // Copy result back from device to host
    cudaMemcpy(y, d_y, n * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_x);
    cudaFree(d_y);
}
