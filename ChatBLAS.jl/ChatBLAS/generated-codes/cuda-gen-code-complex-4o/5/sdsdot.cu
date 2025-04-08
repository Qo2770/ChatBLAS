#include "chatblas_cuda.h"

__global__ void sdsdot_kernel(int n, float b, float *x, float *y, float *res) {
    extern __shared__ double shared_mem[];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int local_tid = threadIdx.x;
    
    double sum = 0.0;
    for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
        sum += (double)x[i] * (double)y[i];
    }

    shared_mem[local_tid] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (local_tid < stride) {
            shared_mem[local_tid] += shared_mem[local_tid + stride];
        }
        __syncthreads();
    }

    if (local_tid == 0) {
        atomicAdd(res, shared_mem[0]);
    }
}

float chatblas_sdsdot(int n, float b, float *x, float *y) {
    float *d_x, *d_y, *d_res;
    float res = 0.0f;

    cudaMalloc((void**)&d_x, n * sizeof(float));
    cudaMalloc((void**)&d_y, n * sizeof(float));
    cudaMalloc((void**)&d_res, sizeof(float));
    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_res, &res, sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    sdsdot_kernel<<<numBlocks, blockSize, blockSize * sizeof(double)>>>(n, b, d_x, d_y, d_res);

    cudaMemcpy(&res, d_res, sizeof(float), cudaMemcpyDeviceToHost);
    res += b;

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_res);

    return res;
}