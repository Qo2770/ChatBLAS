#include "chatblas_cuda.h"

__global__ void dot_product_kernel(int n, float *x, float *y, float *partial_sum) {
    extern __shared__ float cache[];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int cacheIndex = threadIdx.x;
    
    float temp = 0.0f;
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
        partial_sum[blockIdx.x] = cache[0];
    }
}

float chatblas_sdot(int n, float *x, float *y) {
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    float *d_x, *d_y, *d_partial_sum;
    float *h_partial_sum = (float*)malloc(numBlocks * sizeof(float));

    cudaMalloc((void**)&d_x, n * sizeof(float));
    cudaMalloc((void**)&d_y, n * sizeof(float));
    cudaMalloc((void**)&d_partial_sum, numBlocks * sizeof(float));

    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, n * sizeof(float), cudaMemcpyHostToDevice);

    dot_product_kernel<<<numBlocks, blockSize, blockSize * sizeof(float)>>>(n, d_x, d_y, d_partial_sum);

    cudaMemcpy(h_partial_sum, d_partial_sum, numBlocks * sizeof(float), cudaMemcpyDeviceToHost);

    float dot_product = 0.0f;
    for (int i = 0; i < numBlocks; ++i) {
        dot_product += h_partial_sum[i];
    }

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_partial_sum);
    free(h_partial_sum);

    return dot_product;
}
