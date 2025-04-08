#include "chatblas_cuda.h"

__global__ void absolute_sum_kernel(int n, float *x, float *partial_sums) {
    extern __shared__ float shared[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    shared[tid] = (idx < n) ? fabsf(x[idx]) : 0.0f;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial_sums[blockIdx.x] = shared[0];
    }
}

extern "C" float chatblas_sasum(int n, float *x) {
    float *d_x, *d_partial_sums;
    float *partial_sums = (float*)malloc(sizeof(float) * ((n + 255) / 256));
    float total_sum = 0.0f;
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    cudaMalloc((void**)&d_x, n * sizeof(float));
    cudaMalloc((void**)&d_partial_sums, gridSize * sizeof(float));

    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);

    absolute_sum_kernel<<<gridSize, blockSize, blockSize * sizeof(float)>>>(n, d_x, d_partial_sums);

    cudaMemcpy(partial_sums, d_partial_sums, gridSize * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < gridSize; i++) {
        total_sum += partial_sums[i];
    }

    cudaFree(d_x);
    cudaFree(d_partial_sums);
    free(partial_sums);

    return total_sum;
}
