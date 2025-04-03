#include "chatblas_cuda.h"

__global__ void sasum_kernel(int n, float *x, float *sum)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    __shared__ float cache[256];

    float temp = 0;
    while (index < n) {
        temp += fabs(x[index]);
        index += stride;
    }

    cache[threadIdx.x] = temp;
    __syncthreads();

    if (threadIdx.x == 0) {
        float temp = 0;
        for (int i = 0; i < blockDim.x; i++)
            temp += cache[i];
        atomicAdd(sum, temp);
    }
}

float chatblas_sasum(int n, float *x)
{
    float *x_device, *sum_device;
    float sum_host = 0;

    cudaMalloc((void**)&x_device, n * sizeof(float));
    cudaMemcpy(x_device, x, n * sizeof(float), cudaMemcpyHostToDevice);
    
    cudaMalloc((void**)&sum_device, sizeof(float));
    cudaMemcpy(sum_device, &sum_host, sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    sasum_kernel<<<numBlocks, blockSize>>>(n, x_device, sum_device);

    cudaMemcpy(&sum_host, sum_device, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(x_device);
    cudaFree(sum_device);

    return sum_host;
}