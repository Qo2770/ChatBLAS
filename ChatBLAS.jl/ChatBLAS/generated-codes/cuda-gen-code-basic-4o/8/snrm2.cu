#include "chatblas_cuda.h"

__global__ void square_elements(int n, float *x, float *result) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = x[idx] * x[idx];
    }
}

__global__ void reduce_sum(int n, float *input, float *output) {
    extern __shared__ float shared_data[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    shared_data[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && (idx + s < n)) {
            shared_data[tid] += shared_data[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(output, shared_data[0]);
    }
}

float chatblas_snrm2(int n, float *x) {
    if (n <= 0) {
        return 0.0f;
    }

    float *d_x, *d_result, *d_partial_sums;
    float h_result = 0.0f;
    int threads_per_block = 256;
    int blocks = (n + threads_per_block - 1) / threads_per_block;

    cudaMalloc((void**)&d_x, n * sizeof(float));
    cudaMalloc((void**)&d_result, n * sizeof(float));
    cudaMalloc((void**)&d_partial_sums, sizeof(float));
    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_partial_sums, 0, sizeof(float));

    square_elements<<<blocks, threads_per_block>>>(n, d_x, d_result);
    reduce_sum<<<blocks, threads_per_block, threads_per_block * sizeof(float)>>>(n, d_result, d_partial_sums);
    cudaMemcpy(&h_result, d_partial_sums, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_result);
    cudaFree(d_partial_sums);

    return sqrt(h_result);
}
