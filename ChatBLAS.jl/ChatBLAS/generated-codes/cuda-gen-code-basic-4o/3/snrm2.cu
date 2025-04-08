#include "chatblas_cuda.h"

__global__ void sumSquaresKernel(float *x, float *result, int n) {
    extern __shared__ float sharedData[];
    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + tid;
    if (index < n) {
        sharedData[tid] = x[index] * x[index];
    } else {
        sharedData[tid] = 0.0f;
    }
    __syncthreads();

    // Perform reduction to sum up the squares
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sharedData[tid] += sharedData[tid + stride];
        }
        __syncthreads();
    }

    // Write the result of this block to the result array
    if (tid == 0) {
        result[blockIdx.x] = sharedData[0];
    }
}

float chatblas_snrm2(int n, float *x) {
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    float *d_x, *d_partialResults;
    float *h_partialResults = (float *)malloc(numBlocks * sizeof(float));
    float norm = 0.0f;

    cudaMalloc((void **)&d_x, n * sizeof(float));
    cudaMalloc((void **)&d_partialResults, numBlocks * sizeof(float));
    
    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);

    sumSquaresKernel<<<numBlocks, blockSize, blockSize * sizeof(float)>>>(d_x, d_partialResults, n);

    cudaMemcpy(h_partialResults, d_partialResults, numBlocks * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < numBlocks; i++) {
        norm += h_partialResults[i];
    }

    norm = sqrtf(norm);

    cudaFree(d_x);
    cudaFree(d_partialResults);
    free(h_partialResults);

    return norm;
}
