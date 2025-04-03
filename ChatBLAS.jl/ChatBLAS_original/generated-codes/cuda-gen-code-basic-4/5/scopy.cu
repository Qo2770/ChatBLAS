#include "chatblas_cuda.h"

// CUDA Kernel function to copy the values of vector x into y 
__global__ void vecCopy(int n, float *x, float *y) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i < n)
        y[i] = x[i];
}

// Function to copy vector x into y
void chatblas_scopy(int n, float *x, float *y) {
    float *d_x, *d_y;
    int size = n * sizeof(float);
 
    // Allocate device memory for x
    cudaMalloc(&d_x, size);
  
    // Copy vector x to device memory
    cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice);

    // Allocate device memory for y
    cudaMalloc(&d_y, size);
  
    // Copy vector y to device memory
    cudaMemcpy(d_y, y, size, cudaMemcpyHostToDevice);

    // Set grid and block dimensions
    int blockSize = 256;
    int gridSize = ((n + blockSize - 1) / blockSize);
  
    // Launch vecCopy() kernel on GPU
    vecCopy<<<gridSize, blockSize>>>(n, d_x, d_y);

    // Copy result back to host
    cudaMemcpy(y, d_y, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_x);
    cudaFree(d_y); 
}
