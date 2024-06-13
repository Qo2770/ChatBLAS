#include "chatblas_cuda.h"

__global__ void sasum_kernel(int n, float *x, float *sum) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    float local_sum = 0.0f;

    for (int i = tid; i < n; i += stride) {
        local_sum += fabsf(x[i]);
    }

    atomicAdd(sum, local_sum);
}

float chatblas_sasum(int n, float *x) {
    float *d_x, *d_sum, sum;

    // Allocate memory for GPU vectors
    cudaMalloc((void**)&d_x, n * sizeof(float));
    cudaMalloc((void**)&d_sum, sizeof(float));

    // Copy input vector from CPU to GPU
    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);

    // Set initial sum to 0.0
    sum = 0.0f;
    cudaMemcpy(d_sum, &sum, sizeof(float), cudaMemcpyHostToDevice);

    // Define the number of threads per block and the number of blocks
    int threadsPerBlock = 256;
    int numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the kernel
    sasum_kernel<<<numBlocks, threadsPerBlock>>>(n, d_x, d_sum);

    // Copy the sum from GPU to CPU
    cudaMemcpy(&sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_x);
    cudaFree(d_sum);

    return sum;
}