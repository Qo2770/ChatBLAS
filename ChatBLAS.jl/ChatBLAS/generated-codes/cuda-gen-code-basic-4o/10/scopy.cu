#include "chatblas_cuda.h"
#include <cuda_runtime.h>
#include <stdio.h>

// CUDA kernel to copy elements from x to y
__global__ void copyKernel(int n, float *x, float *y) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        y[index] = x[index];
    }
}

void chatblas_scopy(int n, float *x, float *y) {
    if (n <= 0) return;

    float *d_x, *d_y;
    size_t size = n * sizeof(float);

    // Allocate device memory
    cudaMalloc((void**)&d_x, size);
    cudaMalloc((void**)&d_y, size);

    // Copy data from host to device
    cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice);

    // Define block and grid sizes
    int blockSize = 256; // number of threads per block
    int numBlocks = (n + blockSize - 1) / blockSize; // ceiling division

    // Launch kernel on the GPU
    copyKernel<<<numBlocks, blockSize>>>(n, d_x, d_y);

    // Copy the result from device to host
    cudaMemcpy(y, d_y, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_x);
    cudaFree(d_y);
}

int main() {
    int n = 1000;  // Example size
    float *x = (float *)malloc(n * sizeof(float));
    float *y = (float *)malloc(n * sizeof(float));
    
    // Initialize x with some values
    for (int i = 0; i < n; i++) {
        x[i] = static_cast<float>(i);
    }

    // Copy vector x to vector y
    chatblas_scopy(n, x, y);

    // Check if the vectors have been copied correctly
    for (int i = 0; i < n; i++) {
        if (x[i] != y[i]) {
            printf("Error: Mismatch at index %d\n", i);
            return 1;
        }
    }

    printf("Vector copied successfully.\n");

    // Free host memory
    free(x);
    free(y);

    return 0;
}
