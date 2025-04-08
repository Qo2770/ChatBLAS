#include "chatblas_cuda.h"

__global__ void sdsdot_kernel( int n, float b, float *x, float *y, float *res ) {
    __shared__ double cache[256];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;
    
    double temp = 0.0;
    while (tid < n) {
        temp += (double)x[tid] * (double)y[tid];
        tid += blockDim.x * gridDim.x;
    }

    cache[cacheIndex] = temp;

    __syncthreads();

    int i = blockDim.x / 2;
    while (i != 0) {
        if (cacheIndex < i)
            cache[cacheIndex] += cache[cacheIndex + i];
        __syncthreads();
        i /= 2;
    }

    if (cacheIndex == 0) {
        atomicAdd(res, cache[0] + (double)b);
    }
}

float chatblas_sdsdot( int n, float b, float *x, float *y) {
    float *d_x, *d_y;
    float *d_res, h_res;

    cudaMalloc((void**)&d_x, n * sizeof(float));
    cudaMalloc((void**)&d_y, n * sizeof(float));
    cudaMalloc((void**)&d_res, sizeof(float));

    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, n * sizeof(float), cudaMemcpyHostToDevice);

    float initial = 0.0f;
    cudaMemcpy(d_res, &initial, sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    sdsdot_kernel<<<numBlocks, blockSize>>>(n, b, d_x, d_y, d_res);
    cudaDeviceSynchronize();

    cudaMemcpy(&h_res, d_res, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_res);

    return h_res;
}