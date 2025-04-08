#include "chatblas_cuda.h"

// CUDA kernel to compute partial dot products
__global__ void partial_dot_product(int n, float *x, float *y, float *partial_results) {
    __shared__ float shared_data[256];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int tx = threadIdx.x;

    float temp_result = 0.0f;

    while (tid < n) {
        temp_result += x[tid] * y[tid];
        tid += blockDim.x * gridDim.x;
    }

    shared_data[tx] = temp_result;

    __syncthreads();

    // Parallel reduction in shared memory
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tx < stride) {
            shared_data[tx] += shared_data[tx + stride];
        }
        __syncthreads();
    }

    // Write result for this block to global memory
    if (tx == 0) {
        partial_results[blockIdx.x] = shared_data[0];
    }
}

float chatblas_sdot(int n, float *x, float *y) {
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    float *d_x, *d_y, *d_partial_results;
    float *partial_results = (float *)malloc(numBlocks * sizeof(float));
    float dot_product = 0.0f;

    cudaMalloc((void **)&d_x, n * sizeof(float));
    cudaMalloc((void **)&d_y, n * sizeof(float));
    cudaMalloc((void **)&d_partial_results, numBlocks * sizeof(float));

    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, n * sizeof(float), cudaMemcpyHostToDevice);

    partial_dot_product<<<numBlocks, blockSize>>>(n, d_x, d_y, d_partial_results);

    cudaMemcpy(partial_results, d_partial_results, numBlocks * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < numBlocks; ++i) {
        dot_product += partial_results[i];
    }

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_partial_results);
    free(partial_results);

    return dot_product;
}
