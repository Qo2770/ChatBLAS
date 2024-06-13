#include "chatblas_hip.h"

__global__ void sdot_kernel(int n, float *x, float *y, float *res)
{
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? x[i] * y[i] : 0;

    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) 
    {
        if (tid < s) 
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if(tid == 0)
    {
        atomicAdd(res, sdata[0]);
    }
}

float chatblas_sdot( int n, float *x, float *y)
{
    float *dx, *dy, *dr, result;

    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    int sharedMemSize = blockSize * sizeof(float);

    hipMalloc((void**)&dx, n * sizeof(float));
    hipMalloc((void**)&dy, n * sizeof(float));
    hipMalloc((void**)&dr, sizeof(float));

    hipMemcpy(dx, x, n * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(dy, y, n * sizeof(float), hipMemcpyHostToDevice);
    hipMemset(dr, 0, sizeof(float));

    hipLaunchKernelGGL(sdot_kernel, dim3(gridSize), dim3(blockSize), sharedMemSize, 0, n, dx, dy, dr);

    hipMemcpy(&result, dr, sizeof(float), hipMemcpyDeviceToHost);

    hipFree(dx);
    hipFree(dy);
    hipFree(dr);

    return result;
}