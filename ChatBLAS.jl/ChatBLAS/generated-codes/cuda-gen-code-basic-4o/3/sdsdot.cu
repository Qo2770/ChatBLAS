#include "chatblas_cuda.h"
#include <cuda_runtime.h>
#include <stdio.h>

// CUDA kernel to compute partial dot product
__global__ void dotProductKernel(float *x, float *y, double *partialSum, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    double sum = 0.0;
    for (int i = tid; i < n; i += stride) {
        sum += (double)x[i] * (double)y[i];
    }
    partialSum[tid] = sum;
}

// Function to compute the dot product and add scalar b
float chatblas_sdsdot(int n, float b, float *x, float *y) {
    // Allocate device memory
    float *d_x, *d_y;
    double *d_partialSum;
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    cudaMalloc((void**)&d_x, n * sizeof(float));
    cudaMalloc((void**)&d_y, n * sizeof(float));
    cudaMalloc((void**)&d_partialSum, blocksPerGrid * threadsPerBlock * sizeof(double));

    // Copy vectors from host to device
    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, n * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel to compute partial dot products
    dotProductKernel<<<blocksPerGrid, threadsPerBlock>>>(d_x, d_y, d_partialSum, n);

    // Allocate memory for partial sums on the host
    double *h_partialSum = (double*)malloc(blocksPerGrid * threadsPerBlock * sizeof(double));

    // Copy partial sums from device to host
    cudaMemcpy(h_partialSum, d_partialSum, blocksPerGrid * threadsPerBlock * sizeof(double), cudaMemcpyDeviceToHost);

    // Accumulate the partial sums on the host
    double fullSum = 0.0;
    for (int i = 0; i < blocksPerGrid * threadsPerBlock; ++i) {
        fullSum += h_partialSum[i];
    }

    // Add scalar b to the accumulated dot product
    fullSum += (double)b;

    // Free device memory
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_partialSum);
    free(h_partialSum);

    // Return the final result as a float
    return (float)fullSum;
}
