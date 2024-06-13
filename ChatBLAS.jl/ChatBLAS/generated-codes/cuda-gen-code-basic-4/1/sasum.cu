#include "chatblas_cuda.h"
#include <stdio.h>
#include <math.h>

// Kerel function to compute absolute sum
__global__ void abs_sum(const float *x, float *y, int n)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n)
        atomicAdd(y, abs(x[i]));
}

// ChatBlas SASUM function
float chatblas_sasum(int n, float *x)
{
    float *dev_x, *dev_y;
    float y = 0.0;

    // Move data to device
    cudaMalloc((void **)&dev_x, n * sizeof(float));
    cudaMemcpy(dev_x, x, n * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc((void **)&dev_y, sizeof(float));
    cudaMemcpy(dev_y, &y, sizeof(float), cudaMemcpyHostToDevice);

    // Configure and launch the kernel
    dim3 dimBlock(256);
    dim3 dimGrid((n + dimBlock.x - 1)/dimBlock.x);

    abs_sum<<<dimGrid, dimBlock>>>(dev_x, dev_y, n);

    // Move result back to host
    cudaMemcpy(&y, dev_y, sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(dev_x);
    cudaFree(dev_y);

    return y;
}
