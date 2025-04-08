#include "chatblas_cuda.h"

__global__ void computeSquaredSum(int n, float *x, float *result) {
    extern __shared__ float sharedData[];

    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data into shared memory and compute the square of each element
    if (index < n) {
        sharedData[tid] = x[index] * x[index];
    } else {
        sharedData[tid] = 0.0f;
    }
    __syncthreads();

    // Perform reduction to compute the sum of squares
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sharedData[tid] += sharedData[tid + stride];
        }
        __syncthreads();
    }

    // Sum of squares for this block
    if (tid == 0) {
        atomicAdd(result, sharedData[0]);
    }
}

float chatblas_snrm2(int n, float *x) {
    if (n <= 0) {
        return 0.0f;
    }

    float *d_x;
    float *d_result;
    float h_result = 0.0f;

    // Allocate memory on the device
    cudaMalloc((void**)&d_x, n * sizeof(float));
    cudaMalloc((void**)&d_result, sizeof(float));
    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_result, &h_result, sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    // Launch kernel
    computeSquaredSum<<<numBlocks, blockSize, blockSize * sizeof(float)>>>(n, d_x, d_result);

    // Copy result back to host
    cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);

    // Free memory
    cudaFree(d_x);
    cudaFree(d_result);

    // Return the square root of the sum of squares
    return sqrtf(h_result);
}
