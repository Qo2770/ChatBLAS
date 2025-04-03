// chatblas_cuda.h
#ifndef CHATBLAS_CUDA_H
#define CHATBLAS_CUDA_H

#include <cuda_runtime.h>

// CUDA kernel
__global__ void cudaVecScaleKernel(float *dev_x, int n, float a)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        dev_x[i] *= a;
}

#endif // CHATBLAS_CUDA_H
// Include CUDA specific header
#include "chatblas_cuda.h"

void chatblas_sscal( int n, float a , float *x)
{
    // Declare pointer for device memory
    float *dev_x;

    // Size of the array in bytes
    size_t size = n * sizeof(float);

    // Allocate device memory
    cudaMalloc((void**)&dev_x, size);

    // Copy host memory to device memory
    cudaMemcpy(dev_x, x, size, cudaMemcpyHostToDevice);

    // Define block and grid sizes
    int threadsPerBlock = 256;
    int blocksPerGrid =(n + threadsPerBlock - 1) / threadsPerBlock;

    // Execute the kernel
    cudaVecScaleKernel<<<blocksPerGrid, threadsPerBlock>>>(dev_x, n, a);

    // Copy result from device memory to host memory
    cudaMemcpy(x, dev_x, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(dev_x);
}
