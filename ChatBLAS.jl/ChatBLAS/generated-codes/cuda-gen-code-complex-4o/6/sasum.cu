#include "chatblas_cuda.h"

__global__ void sasum_kernel(int n, float *x, float *sum) {
    extern __shared__ float shared_sum[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float local_sum = 0;
    
    while (idx < n) {
        local_sum += fabsf(x[idx]);
        idx += blockDim.x * gridDim.x;
    }

    shared_sum[tid] = local_sum;
    __syncthreads();

    if (blockDim.x >= 512) { if (tid < 256) { shared_sum[tid] += shared_sum[tid + 256]; } __syncthreads(); }
    if (blockDim.x >= 256) { if (tid < 128) { shared_sum[tid] += shared_sum[tid + 128]; } __syncthreads(); }
    if (blockDim.x >= 128) { if (tid < 64) { shared_sum[tid] += shared_sum[tid + 64]; } __syncthreads(); }
    if (tid < 32) {
        if (blockDim.x >= 64) shared_sum[tid] += shared_sum[tid + 32];
        if (blockDim.x >= 32) shared_sum[tid] += shared_sum[tid + 16];
        if (blockDim.x >= 16) shared_sum[tid] += shared_sum[tid + 8];
        if (blockDim.x >= 8) shared_sum[tid] += shared_sum[tid + 4];
        if (blockDim.x >= 4) shared_sum[tid] += shared_sum[tid + 2];
        if (blockDim.x >= 2) shared_sum[tid] += shared_sum[tid + 1];
    }

    if (tid == 0) atomicAdd(sum, shared_sum[0]);
}

float chatblas_sasum(int n, float *x) {
    float *d_x, *d_sum, h_sum = 0;
    cudaMalloc((void**)&d_x, n * sizeof(float));
    cudaMalloc((void**)&d_sum, sizeof(float));
    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_sum, 0, sizeof(float));

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    sasum_kernel<<<numBlocks, blockSize, blockSize * sizeof(float)>>>(n, d_x, d_sum);

    cudaMemcpy(&h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_sum);

    return h_sum;
}