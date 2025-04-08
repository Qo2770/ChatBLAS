#include "chatblas_cuda.h"

__global__ void sasum_kernel(int n, float *x, float *sum) {
    extern __shared__ float shared_sum[];
    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    float temp = 0;

    if (index < n) {
        temp = fabsf(x[index]);
    }
    shared_sum[tid] = temp;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(sum, shared_sum[0]);
    }
}

float chatblas_sasum(int n, float *x) {
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    float *d_x, *d_sum;
    float h_sum = 0;

    cudaMalloc((void**)&d_x, n * sizeof(float));
    cudaMalloc((void**)&d_sum, sizeof(float));
    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sum, &h_sum, sizeof(float), cudaMemcpyHostToDevice);

    sasum_kernel<<<numBlocks, blockSize, blockSize * sizeof(float)>>>(n, d_x, d_sum);
    cudaMemcpy(&h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_sum);

    return h_sum;
}