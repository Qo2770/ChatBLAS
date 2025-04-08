#include "chatblas_cuda.h"

__global__ void findMaxAbsIndex(int n, float *x, int *maxIndex) {
    extern __shared__ float sharedAbsValues[];
    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + tid;

    if (index < n) {
        sharedAbsValues[tid] = fabsf(x[index]);
    } else {
        sharedAbsValues[tid] = -1.0f;
    }

    __syncthreads();

    // Perform reduction to find max absolute value and its index
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride && index + stride < n) {
            if (sharedAbsValues[tid] < sharedAbsValues[tid + stride]) {
                sharedAbsValues[tid] = sharedAbsValues[tid + stride];
                index = index + stride;
            }
        }
        __syncthreads();
    }

    // Write the result for this block to global memory
    if (tid == 0) {
        maxIndex[blockIdx.x] = index;
    }
}

int chatblas_isamax(int n, float *x) {
    if (n < 1) return -1;

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    int *d_maxIndex, *h_maxIndex;
    cudaMalloc((void **)&d_maxIndex, numBlocks * sizeof(int));
    h_maxIndex = (int *)malloc(numBlocks * sizeof(int));

    findMaxAbsIndex<<<numBlocks, blockSize, blockSize * sizeof(float)>>>(n, x, d_maxIndex);

    cudaMemcpy(h_maxIndex, d_maxIndex, numBlocks * sizeof(int), cudaMemcpyDeviceToHost);

    // Find the max index among block results
    int finalIndex = h_maxIndex[0];
    for (int i = 1; i < numBlocks; ++i) {
        if (fabsf(x[h_maxIndex[i]]) > fabsf(x[finalIndex])) {
            finalIndex = h_maxIndex[i];
        }
    }

    cudaFree(d_maxIndex);
    free(h_maxIndex);

    return finalIndex;
}
