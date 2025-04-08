#include "chatblas_cuda.h"

__global__ void sasum_kernel(int n, float *x, float *result) {
    extern __shared__ float shared_data[];
    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + tid;
    float temp_sum = 0.0f;

    // Each thread computes partial sum
    if (index < n)
        shared_data[tid] = fabsf(x[index]);
    else
        shared_data[tid] = 0.0f;
    __syncthreads();

    // Reduce the partial sums in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            shared_data[tid] += shared_data[tid + s];
        __syncthreads();
    }

    // The first thread inside the block will add the partial result to the global result
    if (tid == 0) atomicAdd(result, shared_data[0]);
}

float chatblas_sasum(int n, float *x) {
    float *d_x, *d_result;
    float result = 0.0f;

    // Allocate memory on the device
    cudaMalloc((void **)&d_x, n * sizeof(float));
    cudaMalloc((void **)&d_result, sizeof(float));
    
    // Copy data to device
    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_result, &result, sizeof(float), cudaMemcpyHostToDevice);
    
    int blockSize = 256; // You can adjust this value based on your GPU
    int gridSize = (n + blockSize - 1) / blockSize;

    // Launch kernel
    sasum_kernel<<<gridSize, blockSize, blockSize * sizeof(float)>>>(n, d_x, d_result);
    
    // Copy result back to host
    cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_x);
    cudaFree(d_result);

    return result;
}
