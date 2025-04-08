#include "chatblas_cuda.h"

__global__ void snrm2_kernel(int n, float *x, float *res) {
    extern __shared__ float cache[];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;

    float temp = 0.0;
    while (tid < n) {
        temp += x[tid] * x[tid];
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
        atomicAdd(res, cache[0]);
    }
}

float chatblas_snrm2(int n, float *x) {
    float *d_x, *d_res;
    float result;
    
    cudaMalloc((void**)&d_x, n * sizeof(float));
    cudaMalloc((void**)&d_res, sizeof(float));
    
    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_res, 0, sizeof(float));

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    snrm2_kernel<<<numBlocks, blockSize, blockSize * sizeof(float)>>>(n, d_x, d_res);

    cudaMemcpy(&result, d_res, sizeof(float), cudaMemcpyDeviceToHost);
    
    result = sqrt(result);

    cudaFree(d_x);
    cudaFree(d_res);

    return result;
}