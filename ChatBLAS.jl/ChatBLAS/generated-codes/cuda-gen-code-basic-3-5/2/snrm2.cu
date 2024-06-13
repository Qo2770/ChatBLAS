#include "chatblas_cuda.h"

__global__ void norm_kernel(int n, float *x, float *result) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    // Each thread computes a partial sum
    float sum = 0.0;
    for (int i = tid; i < n; i += stride) {
        sum += x[i] * x[i];
    }

    // Store the partial sum in shared memory
    __shared__ float partial_sums[256];
    partial_sums[threadIdx.x] = sum;
    __syncthreads();

    // Reduce the partial sums in parallel
    int i = blockDim.x/2;
    while (i != 0) {
        if (threadIdx.x < i) {
            partial_sums[threadIdx.x] += partial_sums[threadIdx.x + i];
        }
        __syncthreads();
        i /= 2;
    }

    // Store the final result in global memory
    if (threadIdx.x == 0) {
        atomicAdd(result, partial_sums[0]);
    }
}

float chatblas_snrm2(int n, float *x) {
    float *device_x, *device_result;
    float result = 0.0;

    size_t size = n * sizeof(float);

    // Allocate device memory
    cudaMalloc((void**)&device_x, size);
    cudaMalloc((void**)&device_result, sizeof(float));

    // Copy input vector to device
    cudaMemcpy(device_x, x, size, cudaMemcpyHostToDevice);

    // Launch norm kernel on the GPU
    int threads_per_block = 256;
    int blocks_per_grid = (n + threads_per_block - 1) / threads_per_block;
    norm_kernel<<<blocks_per_grid, threads_per_block>>>(n, device_x, device_result);

    // Copy result back to host
    cudaMemcpy(&result, device_result, sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(device_x);
    cudaFree(device_result);

    // Calculate the square root of the sum
    result = sqrt(result);

    return result;
}
