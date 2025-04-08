#include "chatblas_cuda.h"

__global__ void sdsdot_kernel( int n, float b, float *x, float *y, float *res ) {
    extern __shared__ double shared_mem[];
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int stride = gridDim.x * blockDim.x;
    double dot_product = 0.0;

    for (int i = tid; i < n; i += stride) {
        dot_product += (double)x[i] * (double)y[i];
    }

    shared_mem[threadIdx.x] = dot_product;

    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_mem[threadIdx.x] += shared_mem[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(res, shared_mem[0]);
    }
}

float chatblas_sdsdot( int n, float b, float *x, float *y) {
    float *d_x, *d_y, *d_res;
    float result = 0.0f;
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    cudaMalloc(&d_x, n * sizeof(float));
    cudaMalloc(&d_y, n * sizeof(float));
    cudaMalloc(&d_res, sizeof(float));

    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_res, &result, sizeof(float), cudaMemcpyHostToDevice);

    size_t sharedMemSize = blockSize * sizeof(double);

    sdsdot_kernel<<<numBlocks, blockSize, sharedMemSize>>>(n, b, d_x, d_y, d_res);

    cudaMemcpy(&result, d_res, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_res);

    return result + b;
}