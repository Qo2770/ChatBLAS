#include "chatblas_cuda.h"
#include <cuda_runtime.h>
#include <stdio.h>

// CUDA kernel to compute element-wise product of vectors x and y
__global__ void elementwiseProduct(float *x, float *y, double *partialSum, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < n) {
        // Cast elements to double for accumulation
        double prod = (double)x[tid] * (double)y[tid];
        atomicAdd(partialSum, prod); // Accumulate result using atomic addition
    }
}

float chatblas_sdsdot(int n, float b, float *x, float *y) {
    // Allocate device memory
    float *d_x, *d_y;
    double *d_partialSum;
    double h_partialSum = 0.0; // Host side partial sum

    cudaMalloc((void**)&d_x, n * sizeof(float));
    cudaMalloc((void**)&d_y, n * sizeof(float));
    cudaMalloc((void**)&d_partialSum, sizeof(double));

    // Copy input data to device
    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_partialSum, &h_partialSum, sizeof(double), cudaMemcpyHostToDevice);

    // CUDA kernel launch parameters
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the kernel
    elementwiseProduct<<<blocksPerGrid, threadsPerBlock>>>(d_x, d_y, d_partialSum, n);

    // Copy the result back to host
    cudaMemcpy(&h_partialSum, d_partialSum, sizeof(double), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_partialSum);

    // Add the scalar b to the accumulated dot product and return
    return (float)h_partialSum + b;
}
