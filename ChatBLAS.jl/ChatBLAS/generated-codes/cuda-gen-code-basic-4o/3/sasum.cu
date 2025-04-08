#include "chatblas_cuda.h"

__global__ void compute_abs_sum(float *x, float *partial_sums, int n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

    // Load elements into shared memory and compute absolute value
    sdata[tid] = (index < n) ? fabsf(x[index]) : 0.0f;
    __syncthreads();

    // Perform parallel reduction to compute sum of absolute values
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write result of this block to the partial sums array
    if (tid == 0) {
        partial_sums[blockIdx.x] = sdata[0];
    }
}

float chatblas_sasum(int n, float *x) {
    int blockSize = 256; // Number of threads per block
    int numBlocks = (n + blockSize - 1) / blockSize;

    float *d_x, *d_partial_sums;
    float *partial_sums = (float *)malloc(numBlocks * sizeof(float));
    float abs_sum = 0.0f;

    // Allocate memory on device
    cudaMalloc((void **)&d_x, n * sizeof(float));
    cudaMalloc((void **)&d_partial_sums, numBlocks * sizeof(float));

    // Copy vector to device
    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel to compute partial sums
    compute_abs_sum<<<numBlocks, blockSize, blockSize * sizeof(float)>>>(d_x, d_partial_sums, n);

    // Copy partial sums back to host and compute final sum
    cudaMemcpy(partial_sums, d_partial_sums, numBlocks * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < numBlocks; i++) {
        abs_sum += partial_sums[i];
    }

    // Free device memory
    cudaFree(d_x);
    cudaFree(d_partial_sums);

    // Free host memory
    free(partial_sums);

    return abs_sum;
}
