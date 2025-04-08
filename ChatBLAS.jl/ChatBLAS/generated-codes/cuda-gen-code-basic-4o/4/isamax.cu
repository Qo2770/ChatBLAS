#include "chatblas_cuda.h"

__global__ void findMaxAbsKernel(int n, float *x, int *maxIndex, float *maxValue) {
    extern __shared__ float sharedData[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    sharedData[tid] = (i < n) ? fabsf(x[i]) : 0.0f;
    sharedData[blockDim.x + tid] = i;

    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && sharedData[tid] < sharedData[tid + s]) {
            sharedData[tid] = sharedData[tid + s];
            sharedData[blockDim.x + tid] = sharedData[blockDim.x + tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        if (sharedData[0] > *maxValue) {
            *maxValue = sharedData[0];
            *maxIndex = (int)sharedData[blockDim.x];
        }
    }
}

int chatblas_isamax(int n, float *x) {
    if (n <= 0) return -1;

    int maxIndex = -1;
    float maxValue = 0.0f;
    float *d_x;
    int *d_maxIndex;
    float *d_maxValue;

    cudaMalloc((void**)&d_x, n * sizeof(float));
    cudaMalloc((void**)&d_maxIndex, sizeof(int));
    cudaMalloc((void**)&d_maxValue, sizeof(float));
    
    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_maxValue, &maxValue, sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    findMaxAbsKernel<<<gridSize, blockSize, 2 * blockSize * sizeof(float)>>>(n, d_x, d_maxIndex, d_maxValue);

    cudaMemcpy(&maxIndex, d_maxIndex, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_maxIndex);
    cudaFree(d_maxValue);

    return maxIndex;
}
