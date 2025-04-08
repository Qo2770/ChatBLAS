#include <cuda_runtime.h>
#include "chatblas_cuda.h"

// Kernel to compute the dot product in parallel
__global__ void dotProductKernel(double *d_result, float *d_x, float *d_y, int n) {
    extern __shared__ double sdata[];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    double sum = 0.0;
    for (int i = tid; i < n; i += stride) {
        sum += (double)d_x[i] * (double)d_y[i];
    }

    sdata[threadIdx.x] = sum;
    __syncthreads();

    // Reduce within the block
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    // Result of this block goes to global memory
    if (threadIdx.x == 0) {
        atomicAdd(d_result, sdata[0]);
    }
}

// Function to compute the dot product and add scalar b
float chatblas_sdsdot(int n, float b, float *x, float *y) {
    // Device pointers
    float *d_x = nullptr;
    float *d_y = nullptr;
    double *d_result = nullptr;
    double result = 0;

    // Allocate memory on the device
    cudaMalloc((void **)&d_x, n * sizeof(float));
    cudaMalloc((void **)&d_y, n * sizeof(float));
    cudaMalloc((void **)&d_result, sizeof(double));

    // Copy input vectors from host to device
    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_result, 0, sizeof(double));

    // Execution configuration
    int blockSize = 256; // Number of threads per block
    int gridSize = (n + blockSize - 1) / blockSize; // Number of blocks

    // Launch the kernel
    dotProductKernel<<<gridSize, blockSize, blockSize * sizeof(double)>>>(d_result, d_x, d_y, n);

    // Copy the result back to host
    cudaMemcpy(&result, d_result, sizeof(double), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_result);

    // Add scalar b and cast to float
    return (float)(result + (double)b);
}
