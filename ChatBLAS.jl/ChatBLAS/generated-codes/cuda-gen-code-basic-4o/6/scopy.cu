#include "chatblas_cuda.h"
#include <cuda_runtime.h>
#include <stdio.h>

// CUDA Kernel function to copy vector x to vector y
__global__ void vectorCopyKernel(float *x, float *y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = x[idx];
    }
}

// Host function to manage the vector copy
void chatblas_scopy(int n, float *x, float *y) {
    float *d_x, *d_y; // Device pointers

    // Allocate device memory
    cudaMalloc((void **)&d_x, n * sizeof(float));
    cudaMalloc((void **)&d_y, n * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);

    // Determine block and grid sizes
    int blockSize = 256; // Number of threads per block
    int gridSize = (n + blockSize - 1) / blockSize; // Number of blocks needed

    // Launch kernel
    vectorCopyKernel<<<gridSize, blockSize>>>(d_x, d_y, n);

    // Copy result back to host
    cudaMemcpy(y, d_y, n * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_x);
    cudaFree(d_y);
}

int main() {
    int n = 1000; // Example size
    float *h_x, *h_y;

    // Allocate host memory
    h_x = (float *)malloc(n * sizeof(float));
    h_y = (float *)malloc(n * sizeof(float));

    // Initialize input vector
    for (int i = 0; i < n; i++) {
        h_x[i] = i;
    }

    // Call the vector copy function
    chatblas_scopy(n, h_x, h_y);

    // Verify the result
    for (int i = 0; i < n; i++) {
        if (h_y[i] != h_x[i]) {
            printf("Mismatch at index %d: %f != %f\n", i, h_y[i], h_x[i]);
            break;
        }
    }

    // Free host memory
    free(h_x);
    free(h_y);

    return 0;
}
