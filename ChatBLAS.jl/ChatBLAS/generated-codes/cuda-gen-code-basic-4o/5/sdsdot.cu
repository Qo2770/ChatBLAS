#include "chatblas_cuda.h"
#include <cuda_runtime.h>
#include <stdio.h>

#define THREADS_PER_BLOCK 256

// CUDA kernel for computing part of the dot product
__global__ void dotProductKernel(int n, double *x, double *y, double *partialDot) {
    __shared__ double cache[THREADS_PER_BLOCK];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;
    double temp = 0.0;

    // Each thread computes a part of the dot product
    while (tid < n) {
        temp += x[tid] * y[tid];
        tid += blockDim.x * gridDim.x;
    }

    // Store the partial result in cache
    cache[cacheIndex] = temp;

    // Synchronize threads within the block
    __syncthreads();

    // Reduce the cache to calculate per-block sum
    int i = blockDim.x / 2;
    while (i != 0) {
        if (cacheIndex < i) {
            cache[cacheIndex] += cache[cacheIndex + i];
        }
        __syncthreads();
        i /= 2;
    }

    // Store the per-block result in the partialDot array
    if (cacheIndex == 0) {
        partialDot[blockIdx.x] = cache[0];
    }
}

float chatblas_sdsdot(int n, float b, float *x, float *y) {
    double *d_x, *d_y;
    double *d_partialDot;
    double *h_partialDot;
    int blocks, threads;
    double dotResult = 0.0;

    // Allocate device memory
    cudaMalloc((void**)&d_x, n * sizeof(double));
    cudaMalloc((void**)&d_y, n * sizeof(double));

    // Allocate memory for partial dot products
    blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    cudaMalloc((void**)&d_partialDot, blocks * sizeof(double));
    h_partialDot = (double*)malloc(blocks * sizeof(double));

    // Cast and copy vectors from host to device
    double *h_x = (double*)malloc(n * sizeof(double));
    double *h_y = (double*)malloc(n * sizeof(double));
    for (int i = 0; i < n; i++) {
        h_x[i] = (double)x[i];
        h_y[i] = (double)y[i];
    }
    cudaMemcpy(d_x, h_x, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, n * sizeof(double), cudaMemcpyHostToDevice);

    // Launch kernel to compute dot product
    threads = THREADS_PER_BLOCK;
    dotProductKernel<<<blocks, threads>>>(n, d_x, d_y, d_partialDot);

    // Copy partial results back to host
    cudaMemcpy(h_partialDot, d_partialDot, blocks * sizeof(double), cudaMemcpyDeviceToHost);

    // Accumulate results
    for (int i = 0; i < blocks; ++i) {
        dotResult += h_partialDot[i];
    }

    // Add scalar b to the dot result and cast back to float
    float result = (float)(dotResult + (double)b);

    // Clean up
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_partialDot);
    free(h_x);
    free(h_y);
    free(h_partialDot);

    return result;
}
