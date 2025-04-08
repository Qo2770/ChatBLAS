#include "chatblas_cuda.h"
#include <cuda_runtime.h>
#include <stdio.h>

// CUDA kernel for computing partial dot product in parallel
__global__ void dotProductKernel(int n, const float *x, const float *y, double *partialSums) {
    extern __shared__ double sdata[];

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int index = threadIdx.x;
    double sum = 0.0;

    if (tid < n) {
        sum = (double)x[tid] * (double)y[tid];
    }

    sdata[index] = sum;
    __syncthreads();

    // Perform reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (index < s && (index + s) < n) {
            sdata[index] += sdata[index + s];
        }
        __syncthreads();
    }

    // Write the result of this block to partialSums
    if (index == 0) {
        partialSums[blockIdx.x] = sdata[0];
    }
}

// Function to compute dot product and add scalar b
float chatblas_sdsdot(int n, float b, float *x, float *y) {
    const int threadsPerBlock = 256;
    const int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;

    float *d_x = NULL;
    float *d_y = NULL;
    double *d_partialSums = NULL;
    double *h_partialSums = (double *)malloc(blocks * sizeof(double));
    double finalSum = 0.0;

    // Allocate device memory
    cudaMalloc((void **)&d_x, n * sizeof(float));
    cudaMalloc((void **)&d_y, n * sizeof(float));
    cudaMalloc((void **)&d_partialSums, blocks * sizeof(double));

    // Copy vectors from host to device
    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, n * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel with reduction
    dotProductKernel<<<blocks, threadsPerBlock, threadsPerBlock * sizeof(double)>>>(n, d_x, d_y, d_partialSums);

    // Copy back the partial sums from device to host
    cudaMemcpy(h_partialSums, d_partialSums, blocks * sizeof(double), cudaMemcpyDeviceToHost);

    // Sum up the partial sums from each block
    for (int i = 0; i < blocks; i++) {
        finalSum += h_partialSums[i];
    }

    // Add scalar b
    finalSum += (double)b;

    // Free device memory
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_partialSums);

    // Free host memory
    free(h_partialSums);

    // Return the result as a float
    return (float)finalSum;
}
