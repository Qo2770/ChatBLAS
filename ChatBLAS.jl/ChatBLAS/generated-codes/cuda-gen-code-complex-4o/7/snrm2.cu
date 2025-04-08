#include "chatblas_cuda.h"

__global__ void snrm2_kernel(int n, float *x, float *res) {
    extern __shared__ float cache[];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIdx = threadIdx.x;
    
    float temp = 0.0f;
    while (tid < n) {
        temp += x[tid] * x[tid];
        tid += blockDim.x * gridDim.x;
    }
    
    cache[cacheIdx] = temp;
    
    __syncthreads();
    
    int i = blockDim.x / 2;
    while (i != 0) {
        if (cacheIdx < i)
            cache[cacheIdx] += cache[cacheIdx + i];
        __syncthreads();
        i /= 2;
    }
    
    if (cacheIdx == 0) {
        atomicAdd(res, cache[0]);
    }
}

float chatblas_snrm2(int n, float *x) {
    float *d_x, *d_res, h_res = 0.0f;
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    cudaMalloc((void**)&d_x, n * sizeof(float));
    cudaMalloc((void**)&d_res, sizeof(float));
    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_res, &h_res, sizeof(float), cudaMemcpyHostToDevice);
    
    snrm2_kernel<<<numBlocks, blockSize, blockSize * sizeof(float)>>>(n, d_x, d_res);
    cudaMemcpy(&h_res, d_res, sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(d_x);
    cudaFree(d_res);
    
    return sqrtf(h_res);
}