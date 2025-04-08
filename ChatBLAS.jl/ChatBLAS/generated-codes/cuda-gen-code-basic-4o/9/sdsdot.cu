#include "chatblas_cuda.h"
#include <cuda_runtime.h>
#include <stdio.h>

// CUDA kernel to compute partial dot product
__global__ void dotProductKernel(double *result, const float *x, const float *y, int n) {
    extern __shared__ double shared[];

    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + tid;
    
    if (index < n) {
        // Accumulate in shared memory
        shared[tid] = static_cast<double>(x[index]) * static_cast<double>(y[index]);
    } else {
        shared[tid] = 0.0;
    }
    __syncthreads();

    // Perform reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }

    // Write the result for this block to global memory
    if (tid == 0) {
        atomicAdd(result, shared[0]);
    }
}

float chatblas_sdsdot(int n, float b, float *x, float *y) {
    // Device memory pointers
    float *d_x, *d_y;
    double *d_result;
    double h_result = 0.0;

    // Allocate device memory
    cudaMalloc((void**)&d_x, n * sizeof(float));
    cudaMalloc((void**)&d_y, n * sizeof(float));
    cudaMalloc((void**)&d_result, sizeof(double));

    // Copy vectors from host to device
    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_result, &h_result, sizeof(double), cudaMemcpyHostToDevice);

    // Launch dotProductKernel on the GPU
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    dotProductKernel<<<gridSize, blockSize, blockSize * sizeof(double)>>>(d_result, d_x, d_y, n);

    // Copy result back to host
    cudaMemcpy(&h_result, d_result, sizeof(double), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_result);

    // Add the scalar b and return final result
    return static_cast<float>(h_result) + b;
}
