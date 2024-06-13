#include "chatblas_cuda.h"

#define THREADS_PER_BLOCK 256

__global__ void swap_kernel(int n, float *x, float *y)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i+= stride)
    {
        float temp = x[i];
        x[i] = y[i];
        y[i] = temp;
    }
}

void chatblas_sswap(int n, float *x, float *y)
{
    int blocks_per_grid = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    swap_kernel<<<blocks_per_grid, THREADS_PER_BLOCK>>>(n, x, y);
    cudaDeviceSynchronize();
}
