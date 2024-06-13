#include "chatblas_cuda.h"
#include <cublas_v2.h>

__global__ void dotProdKernel(int n, float *x, float *y, float *result) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    __shared__ float temp[MAX_THREADS_PER_BLOCK];

    temp[threadIdx.x] = 0;

    if (index < n)
        temp[threadIdx.x] = x[index] * y[index];

    __syncthreads();

    if (threadIdx.x == 0) {
        float sum = 0;
        for (int i = 0; i < blockDim.x; i++)
            sum += temp[i];

        atomicAdd(result, sum);
    }
}

float chatblas_sdot(int n, float *x, float *y) {
    float *dev_x, *dev_y, *dev_result;
    float result;

    cudaMalloc((void**)&dev_x, n * sizeof(float));
    cudaMalloc((void**)&dev_y, n * sizeof(float));
    cudaMalloc((void**)&dev_result, sizeof(float));

    cudaMemcpy(dev_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_y, y, n * sizeof(float), cudaMemcpyHostToDevice);

    result = 0.0;
    cudaMemcpy(dev_result, &result, sizeof(float), cudaMemcpyHostToDevice);

    dotProdKernel<<<(n + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK, MAX_THREADS_PER_BLOCK>>>(n, dev_x, dev_y, dev_result);

    cudaMemcpy(&result, dev_result, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dev_x);
    cudaFree(dev_y);
    cudaFree(dev_result);

    return result;
}
