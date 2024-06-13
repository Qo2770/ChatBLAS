#include "chatblas_cuda.h"

__global__ void dotProduct(int n, float *x, float *y, float *result) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    float sum = 0;
    for (int i = index; i < n; i += stride) {
        sum += x[i] * y[i];
    }

    atomicAdd(result, sum);
}

float chatblas_sdot(int n, float *x, float *y) {
    float *dev_x, *dev_y, *dev_result;
    float result = 0;

    cudaMalloc((void **)&dev_x, n * sizeof(float));
    cudaMalloc((void **)&dev_y, n * sizeof(float));
    cudaMalloc((void **)&dev_result, sizeof(float));

    cudaMemcpy(dev_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_y, y, n * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    dotProduct<<<numBlocks, blockSize>>>(n, dev_x, dev_y, dev_result);

    cudaMemcpy(&result, dev_result, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dev_x);
    cudaFree(dev_y);
    cudaFree(dev_result);

    return result;
}