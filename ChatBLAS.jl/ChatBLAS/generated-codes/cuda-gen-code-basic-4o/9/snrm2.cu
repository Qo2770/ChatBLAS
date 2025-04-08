#include "chatblas_cuda.h"

__global__ void computePartialSums(int n, float *x, float *partial_sums) {
    extern __shared__ float shared_data[];
    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    float temp = 0.0f;

    if (index < n) {
        temp = x[index] * x[index];
    }
    shared_data[tid] = temp;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial_sums[blockIdx.x] = shared_data[0];
    }
}

float chatblas_snrm2(int n, float *x) {
    float *d_x, *d_partial_sums;
    float h_partial_sums[256];  // Adjust size according to expected number of blocks
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    cudaMalloc((void **)&d_x, n * sizeof(float));
    cudaMalloc((void **)&d_partial_sums, gridSize * sizeof(float));
    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);

    computePartialSums<<<gridSize, blockSize, blockSize * sizeof(float)>>>(n, d_x, d_partial_sums);

    cudaMemcpy(h_partial_sums, d_partial_sums, gridSize * sizeof(float), cudaMemcpyDeviceToHost);

    float sum = 0.0f;
    for (int i = 0; i < gridSize; i++) {
        sum += h_partial_sums[i];
    }

    cudaFree(d_x);
    cudaFree(d_partial_sums);

    return sqrtf(sum);
}
