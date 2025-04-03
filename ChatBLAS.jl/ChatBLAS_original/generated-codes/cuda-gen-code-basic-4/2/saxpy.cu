#include "chatblas_cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void chatblas_saxpy_kernel(int n, float a, float *x, float *y)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n) 
    {
        y[i] = a * x[i] + y[i];
    }
}

void chatblas_saxpy(int n, float a, float *x, float *y)
{
    int threadsPerBlock = 256;
    int blocksPerGrid =(n + threadsPerBlock - 1) / threadsPerBlock;
    chatblas_saxpy_kernel<<<blocksPerGrid, threadsPerBlock>>>(n, a, x, y);

    cudaError err = cudaGetLastError();
    if (err != cudaSuccess) 
    {
        printf("Error: %s\n", cudaGetErrorString(err));
    }
}
