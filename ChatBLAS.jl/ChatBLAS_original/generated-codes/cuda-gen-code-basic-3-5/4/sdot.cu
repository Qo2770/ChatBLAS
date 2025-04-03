#include "chatblas_cuda.h"

__global__ void dotProduct(int n, float *x, float *y, float *result) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    float partialSum = 0.0f;
    for (int i = tid; i < n; i += stride) {
        partialSum += x[i] * y[i];
    }

    atomicAdd(result, partialSum);
}

float chatblas_sdot(int n, float *x, float *y) {
    float *deviceX, *deviceY, *deviceResult;
    float result = 0.0f;

    cudaMalloc((void **)&deviceX, n * sizeof(float));
    cudaMalloc((void **)&deviceY, n * sizeof(float));
    cudaMalloc((void **)&deviceResult, sizeof(float));

    cudaMemcpy(deviceX, x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceY, y, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(deviceResult, 0, sizeof(float));

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    dotProduct<<<numBlocks, blockSize>>>(n, deviceX, deviceY, deviceResult);

    cudaMemcpy(&result, deviceResult, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(deviceX);
    cudaFree(deviceY);
    cudaFree(deviceResult);

    return result;
}
