#include "chatblas_cuda.h"

__global__ void sasum_kernel(int n, float *x, float *result) {
    extern __shared__ float shared_data[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    shared_data[tid] = (i < n) ? fabsf(x[i]) : 0.0f;
    __syncthreads();

    // Reduce within the block
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        __syncthreads();
    }

    // Write the block's result to the global memory
    if (tid == 0) {
        atomicAdd(result, shared_data[0]);
    }
}

float chatblas_sasum(int n, float *x) {
    float *d_x, *d_result;
    float h_result = 0.0f;

    // Allocate memory on the device
    cudaMalloc((void **)&d_x, n * sizeof(float));
    cudaMalloc((void **)&d_result, sizeof(float));
    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_result, &h_result, sizeof(float), cudaMemcpyHostToDevice);

    // Kernel call
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    sasum_kernel<<<gridSize, blockSize, blockSize * sizeof(float)>>>(n, d_x, d_result);

    // Copy the result back to host
    cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_x);
    cudaFree(d_result);

    return h_result;
}
