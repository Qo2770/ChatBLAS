#include "chatblas_cuda.h"

__global__ void dotProductKernel(float *x, float *y, float *result, int n) {
    extern __shared__ float temp[];

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int threadId = threadIdx.x;

    // Load elements into shared memory and compute the partial dot product
    temp[threadId] = (index < n) ? x[index] * y[index] : 0.0f;
    __syncthreads();

    // Perform parallel reduction
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadId < stride)
            temp[threadId] += temp[threadId + stride];
        __syncthreads();
    }

    // Thread 0 writes the result for this block to global memory
    if (threadId == 0)
        atomicAdd(result, temp[0]);
}

float chatblas_sdot(int n, float *x, float *y) {
    float *d_x, *d_y, *d_result;
    float h_result = 0.0f;

    cudaMalloc((void**)&d_x, n * sizeof(float));
    cudaMalloc((void**)&d_y, n * sizeof(float));
    cudaMalloc((void**)&d_result, sizeof(float));

    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_result, &h_result, sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    dotProductKernel<<<gridSize, blockSize, blockSize * sizeof(float)>>>(d_x, d_y, d_result, n);

    cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_result);

    return h_result;
}
