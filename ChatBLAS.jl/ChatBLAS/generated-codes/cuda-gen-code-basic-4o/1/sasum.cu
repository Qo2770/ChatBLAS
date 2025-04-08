#include "chatblas_cuda.h"

__global__ void computeAbsoluteSum(int n, float *x, float *partialSums) {
    extern __shared__ float sharedData[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Each thread loads one element into shared memory
    sharedData[tid] = (i < n) ? fabsf(x[i]) : 0.0f;
    __syncthreads();
    
    // Perform reduction in shared memory to sum the absolute values
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sharedData[tid] += sharedData[tid + s];
        }
        __syncthreads();
    }
    
    // Write the result for this block to the partialSums array
    if (tid == 0) {
        partialSums[blockIdx.x] = sharedData[0];
    }
}

float chatblas_sasum(int n, float *x) {
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    float *d_x, *d_partialSums;
    float *partialSums = (float *)malloc(numBlocks * sizeof(float));
    float totalSum = 0.0f;

    // Allocate memory on the GPU
    cudaMalloc((void **)&d_x, n * sizeof(float));
    cudaMalloc((void **)&d_partialSums, numBlocks * sizeof(float));
    
    // Copy the input data to the GPU
    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
    
    // Launch the kernel
    computeAbsoluteSum<<<numBlocks, blockSize, blockSize * sizeof(float)>>>(n, d_x, d_partialSums);

    // Copy the partial results back to the CPU
    cudaMemcpy(partialSums, d_partialSums, numBlocks * sizeof(float), cudaMemcpyDeviceToHost);

    // Sum up the partial results to get the final total sum
    for (int i = 0; i < numBlocks; i++) {
        totalSum += partialSums[i];
    }
    
    // Free the memory
    cudaFree(d_x);
    cudaFree(d_partialSums);
    free(partialSums);

    return totalSum;
}
