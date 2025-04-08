#include "chatblas_cuda.h"

__global__ void sasum_kernel(int n, float *x, float *sum) {
    extern __shared__ float shared_data[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    float temp_sum = 0;

    while (idx < n) {
        temp_sum += fabsf(x[idx]);
        idx += blockDim.x * gridDim.x;
    }
    shared_data[tid] = temp_sum;
    __syncthreads();

    // Reduce within block
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_data[tid] += shared_data[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(sum, shared_data[0]);
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