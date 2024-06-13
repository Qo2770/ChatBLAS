#include "chatblas_cuda.h"

__global__ void saxpy_kernel(int n, float a, float *x, float *y)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (index < n)
    {
        y[index] = a * x[index] + y[index];
    }
}

void chatblas_saxpy(int n, float a, float *x, float *y)
{
    int size = n * sizeof(float);
    float *d_x, *d_y;
    
    cudaMalloc((void **)&d_x, size);
    cudaMalloc((void **)&d_y, size);
    
    cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, size, cudaMemcpyHostToDevice);
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    saxpy_kernel<<<blocksPerGrid, threadsPerBlock>>>(n, a, d_x, d_y);

    cudaMemcpy(y, d_y, size, cudaMemcpyDeviceToHost);
    
    cudaFree(d_x);
    cudaFree(d_y);
}

