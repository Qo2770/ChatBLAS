#include "chatblas_cuda.h"

// CUDA kernel to scale the vector
__global__ void scaleVectorKernel(int n, float a, float *x) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        x[index] = a * x[index];
    }
}

// Function to scale the vector using CUDA
void chatblas_sscal(int n, float a, float *x) {
    float *d_x;
    size_t size = n * sizeof(float);

    // Allocate memory on the device
    cudaMalloc(&d_x, size);

    // Copy data from host to device
    cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice);

    // Define block and grid sizes
    int blockSize = 256; // Number of threads per block
    int numBlocks = (n + blockSize - 1) / blockSize; // Number of blocks needed

    // Launch the kernel
    scaleVectorKernel<<<numBlocks, blockSize>>>(n, a, d_x);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(x, d_x, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_x);
}
