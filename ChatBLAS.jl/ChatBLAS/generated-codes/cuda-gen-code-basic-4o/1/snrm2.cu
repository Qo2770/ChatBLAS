#include "chatblas_cuda.h"

__global__ void compute_squared_norm(int n, float *x, float *result) {
    extern __shared__ float shared_data[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    shared_data[tid] = (i < n) ? x[i] * x[i] : 0;
    __syncthreads();

    // Reduce within block
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_data[tid] += shared_data[tid + s];
        }
        __syncthreads();
    }

    // Write block result to global memory
    if (tid == 0) {
        atomicAdd(result, shared_data[0]);
    }
}

float chatblas_snrm2(int n, float *x) {
    float *d_x, *d_result;
    float h_result = 0.0f;

    // Allocate device memory
    cudaMalloc((void**)&d_x, n * sizeof(float));
    cudaMalloc((void**)&d_result, sizeof(float));

    // Copy data to device
    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_result, &h_result, sizeof(float), cudaMemcpyHostToDevice);

    // Define block and grid sizes
    int blockSize = 256; // You can adjust this value
    int gridSize = (n + blockSize - 1) / blockSize;

    // Launch kernel
    compute_squared_norm<<<gridSize, blockSize, blockSize * sizeof(float)>>>(n, d_x, d_result);
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_x);
    cudaFree(d_result);

    // Return the square root of the sum of squares to get the Euclidean norm
    return sqrtf(h_result);
}
