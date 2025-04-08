#include "chatblas_cuda.h"

__global__ void dotProductKernel(float *x, float *y, float *partialSum, int n) {
    extern __shared__ float cache[];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;
    
    float temp = 0.0;
    while (tid < n) {
        temp += x[tid] * y[tid];
        tid += blockDim.x * gridDim.x;
    }
    
    cache[cacheIndex] = temp;
    __syncthreads();

    int i = blockDim.x / 2;
    while (i != 0) {
        if (cacheIndex < i) {
            cache[cacheIndex] += cache[cacheIndex + i];
        }
        __syncthreads();
        i /= 2;
    }
    
    if (cacheIndex == 0) {
        partialSum[blockIdx.x] = cache[0];
    }
}

float chatblas_sdot(int n, float *x, float *y) {
    float *d_x, *d_y, *d_partialSum;
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    size_t size = n * sizeof(float);
    size_t partialSize = gridSize * sizeof(float);
    float *partialSum = (float *)malloc(partialSize);

    cudaMalloc((void **)&d_x, size);
    cudaMalloc((void **)&d_y, size);
    cudaMalloc((void **)&d_partialSum, partialSize);

    cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, size, cudaMemcpyHostToDevice);

    dotProductKernel<<<gridSize, blockSize, blockSize * sizeof(float)>>>(d_x, d_y, d_partialSum, n);

    cudaMemcpy(partialSum, d_partialSum, partialSize, cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_partialSum);

    float dotProduct = 0.0;
    for (int i = 0; i < gridSize; i++) {
        dotProduct += partialSum[i];
    }

    free(partialSum);

    return dotProduct;
}
