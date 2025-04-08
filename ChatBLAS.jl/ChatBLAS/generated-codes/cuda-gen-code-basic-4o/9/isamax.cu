#include "chatblas_cuda.h"

__global__ void findMaxAbsIndex(int n, float *x, int *maxIdx, float *maxVal) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        sdata[tid] = fabsf(x[i]);
    } else {
        sdata[tid] = -1.0; // Assuming all x elements are non-negative
    }
    __syncthreads();

    // Perform reduction to find max absolute value
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && i + s < n) {
            if (sdata[tid + s] > sdata[tid]) {
                sdata[tid] = sdata[tid + s];
                // Store the index of the max element
                if (maxIdx != nullptr) {
                    maxIdx[tid] = i + s;
                }
            }
        }
        __syncthreads();
    }

    // Write the result for this block to global memory
    if (tid == 0) {
        maxVal[blockIdx.x] = sdata[0];
        maxIdx[blockIdx.x] = blockIdx.x * blockDim.x;
    }
}

int chatblas_isamax(int n, float *x) {
    if (n <= 0) return -1;

    // Device copies of x, maxIdx and maxVal
    float *d_x;
    int *d_maxIdx;
    float *d_maxVal;

    cudaMalloc((void **)&d_x, n * sizeof(float));
    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);

    int numBlocks = (n + 255) / 256;
    cudaMalloc((void **)&d_maxIdx, numBlocks * sizeof(int));
    cudaMalloc((void **)&d_maxVal, numBlocks * sizeof(float));

    findMaxAbsIndex<<<numBlocks, 256, 256 * sizeof(float)>>>(n, d_x, d_maxIdx, d_maxVal);

    int *h_maxIdx = (int *)malloc(numBlocks * sizeof(int));
    float *h_maxVal = (float *)malloc(numBlocks * sizeof(float));

    cudaMemcpy(h_maxIdx, d_maxIdx, numBlocks * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_maxVal, d_maxVal, numBlocks * sizeof(float), cudaMemcpyDeviceToHost);

    // Find the max value among the blocks
    int maxIndex = 0;
    float maxValue = h_maxVal[0];
    for (int i = 1; i < numBlocks; i++) {
        if (h_maxVal[i] > maxValue) {
            maxValue = h_maxVal[i];
            maxIndex = h_maxIdx[i];
        }
    }

    // Clean up
    cudaFree(d_x);
    cudaFree(d_maxIdx);
    cudaFree(d_maxVal);
    free(h_maxIdx);
    free(h_maxVal);

    return maxIndex;
}
