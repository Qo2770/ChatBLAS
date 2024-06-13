#include "chatblas_cuda.h"
#include <cuda_runtime.h>

__global__ void calculateSum(int n, float *x, float *result) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = abs(x[idx]);
    }
}

float chatblas_sasum(int n, float *x) {
    float *d_x, *d_result, result = 0.0f;
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;

    // Allocate memory on the device
    cudaMalloc((void **)&d_x, n * sizeof(float));
    cudaMalloc((void **)&d_result, n * sizeof(float));

    // Copy input vector to device
    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel to calculate sum of absolute values
    calculateSum<<<grid_size, block_size>>>(n, d_x, d_result);

    // Copy result vector from device to host
    cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_x);
    cudaFree(d_result);

    return result;
}