#include "chatblas_cuda.h"

__global__ void absSum(float *x, int n, float *sum) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    float localSum = 0.0f;

    for (int i = tid; i < n; i += stride) {
        localSum += fabsf(x[i]);
    }

    atomicAdd(sum, localSum);
}

float chatblas_sasum(int n, float *x) {
    float *dev_x, *dev_sum, sum = 0.0f;
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    cudaMalloc((void**)&dev_x, n * sizeof(float));
    cudaMalloc((void**)&dev_sum, sizeof(float));

    cudaMemcpy(dev_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(dev_sum, 0, sizeof(float));

    absSum<<<gridSize, blockSize>>>(dev_x, n, dev_sum);

    cudaMemcpy(&sum, dev_sum, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dev_x);
    cudaFree(dev_sum);

    return sum;
}
