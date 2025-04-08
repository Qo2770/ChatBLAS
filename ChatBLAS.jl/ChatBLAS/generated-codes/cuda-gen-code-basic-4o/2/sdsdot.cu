#include "chatblas_cuda.h"
#include <cuda_runtime.h>
#include <stdio.h>

#define BLOCK_SIZE 256  // Define the number of threads per block

// CUDA kernel to compute partial dot products
__global__ void dotProductKernel(float *x, float *y, double *partialSums, int n) {
    __shared__ double cache[BLOCK_SIZE];  // Shared memory for storing partial results

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIdx = threadIdx.x;

    double temp = 0.0;

    // Accumulate products into temp
    while (tid < n) {
        temp += (double)x[tid] * (double)y[tid];
        tid += blockDim.x * gridDim.x;
    }

    // Store temp result into shared memory
    cache[cacheIdx] = temp;

    // Synchronize threads in the block
    __syncthreads();

    // Reduce within the block (sum up the block's contributions)
    int i = blockDim.x / 2;
    while (i != 0) {
        if (cacheIdx < i) {
            cache[cacheIdx] += cache[cacheIdx + i];
        }
        __syncthreads();
        i /= 2;
    }

    // Store the result of this block to global memory
    if (cacheIdx == 0) {
        partialSums[blockIdx.x] = cache[0];
    }
}

// Host function to compute the dot product and add scalar b
float chatblas_sdsdot(int n, float b, float *x, float *y) {
    float *d_x, *d_y;
    double *d_partialSums;
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Allocate device memory
    cudaMalloc((void **)&d_x, n * sizeof(float));
    cudaMalloc((void **)&d_y, n * sizeof(float));
    cudaMalloc((void **)&d_partialSums, blocks * sizeof(double));

    // Copy vectors from host to device
    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, n * sizeof(float), cudaMemcpyHostToDevice);

    // Launch the kernel
    dotProductKernel<<<blocks, BLOCK_SIZE>>>(d_x, d_y, d_partialSums, n);

    // Allocate partial sums on host and copy from device
    double *partialSums = (double *)malloc(blocks * sizeof(double));
    cudaMemcpy(partialSums, d_partialSums, blocks * sizeof(double), cudaMemcpyDeviceToHost);

    // Sum up the partial results on host
    double dotProductResult = 0.0;
    for (int i = 0; i < blocks; i++) {
        dotProductResult += partialSums[i];
    }

    // Free device memory
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_partialSums);
    free(partialSums);

    // Add scalar b and return the result
    return float(dotProductResult + (double)b);
}
