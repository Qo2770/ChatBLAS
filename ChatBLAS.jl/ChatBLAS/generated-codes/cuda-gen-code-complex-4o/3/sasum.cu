#include "chatblas_cuda.h"

__global__ void sasum_kernel(int n, float *x, float *sum) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float tmp = 0;

    while (i < n) {
        tmp += fabsf(x[i]);
        i += blockDim.x * gridDim.x;
    }
    
    sdata[tid] = tmp;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(sum, sdata[0]);
    }
}

float chatblas_sasum(int n, float *x) {
    float *d_x, *d_sum;
    float h_sum = 0;
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    cudaMalloc((void **)&d_x, n * sizeof(float));
    cudaMalloc((void **)&d_sum, sizeof(float));

    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_sum, 0, sizeof(float));

    sasum_kernel<<<numBlocks, blockSize, blockSize * sizeof(float)>>>(n, d_x, d_sum);
    cudaMemcpy(&h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_sum);

    return h_sum;
}