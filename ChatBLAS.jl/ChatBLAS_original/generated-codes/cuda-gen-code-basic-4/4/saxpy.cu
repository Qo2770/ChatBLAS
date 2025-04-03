#include "chatblas_cuda.h"

// CUDA Kernel for SAXPY
__global__ void saxpy_kernel(int n, float a, float *x, float *y)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Check if within bounds of vector
    if (idx < n)
    {
        y[idx] = a * x[idx] + y[idx];
    }
}


void chatblas_saxpy(int n, float a, float *x, float *y)
{
    // Define the number of blocks and threads per block
    int blocks = (n + 255) / 256;
    int threads_per_block = 256;

    // Memory allocation and copy for the device vectors
    float *d_x, *d_y;
    cudaMalloc((void **)&d_x, n*sizeof(float));
    cudaMalloc((void **)&d_y, n*sizeof(float));
    cudaMemcpy(d_x, x, n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, n*sizeof(float), cudaMemcpyHostToDevice);

    // Launch the kernel on the GPU
    saxpy_kernel<<<blocks, threads_per_block>>>(n, a, d_x, d_y);

    // Copy the result back to host memory
    cudaMemcpy(y, d_y, n*sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_x);
    cudaFree(d_y);
}
