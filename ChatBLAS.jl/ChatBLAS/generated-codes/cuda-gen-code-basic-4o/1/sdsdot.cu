#include "chatblas_cuda.h"
#include <cuda_runtime.h>
#include <stdio.h>

// CUDA kernel to compute element-wise multiplication of vectors
__global__ void dotProductKernel(int n, const float *x, const float *y, double *partialSums) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        partialSums[i] = (double)x[i] * (double)y[i];
    }
}

// Function to compute dot product in parallel using CUDA
float chatblas_sdsdot(int n, float b, float *x, float *y) {
    float *d_x, *d_y;
    double *d_partialSums;
    double result = 0.0;

    // Allocate device memory
    cudaMalloc((void**)&d_x, n * sizeof(float));
    cudaMalloc((void**)&d_y, n * sizeof(float));
    cudaMalloc((void**)&d_partialSums, n * sizeof(double));

    // Copy vectors from host to device
    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, n * sizeof(float), cudaMemcpyHostToDevice);

    // Decide block size and grid size
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    // Launch kernel
    dotProductKernel<<<gridSize, blockSize>>>(n, d_x, d_y, d_partialSums);

    // Allocate host memory for partial sums
    double *partialSums = (double*)malloc(n * sizeof(double));
    if (partialSums == NULL) {
        fprintf(stderr, "Failed to allocate host memory!\n");
        return -1;
    }

    // Copy results from device to host
    cudaMemcpy(partialSums, d_partialSums, n * sizeof(double), cudaMemcpyDeviceToHost);

    // Accumulate results
    for (int i = 0; i < n; ++i) {
        result += partialSums[i];
    }

    // Add scalar b
    result += (double)b;

    // Free device and host memory
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_partialSums);
    free(partialSums);

    return (float)result;
}