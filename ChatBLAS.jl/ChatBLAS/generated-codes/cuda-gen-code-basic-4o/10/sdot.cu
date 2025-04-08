#include "chatblas_cuda.h"

__global__ void dot_product_kernel(int n, float *x, float *y, float *temp) {
    __shared__ float cache[256];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int cacheIndex = threadIdx.x;

    float sum = 0.0;

    // Perform the partial dot product
    while (tid < n) {
        sum += x[tid] * y[tid];
        tid += blockDim.x * gridDim.x;
    }

    // Set the cache values
    cache[cacheIndex] = sum;

    // Synchronize threads in this block
    __syncthreads();

    // Perform reduction within the block
    int i = blockDim.x / 2;
    while (i != 0) {
        if (cacheIndex < i) {
            cache[cacheIndex] += cache[cacheIndex + i];
        }
        __syncthreads();
        i /= 2;
    }

    // Add the sum of this block to the global sum
    if (cacheIndex == 0) {
        atomicAdd(temp, cache[0]);
    }
}

float chatblas_sdot(int n, float *x, float *y) {
    float *d_x, *d_y, *d_temp;
    float h_temp = 0.0;
    cudaMalloc(&d_x, n * sizeof(float));
    cudaMalloc(&d_y, n * sizeof(float));
    cudaMalloc(&d_temp, sizeof(float));

    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_temp, &h_temp, sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    dot_product_kernel<<<gridSize, blockSize>>>(n, d_x, d_y, d_temp);

    cudaMemcpy(&h_temp, d_temp, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_temp);

    return h_temp;
}
