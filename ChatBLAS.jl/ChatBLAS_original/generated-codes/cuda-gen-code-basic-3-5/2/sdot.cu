#include "chatblas_cuda.h"

__global__ void dotProduct(int n, float *x, float *y, float *result)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    __shared__ float partialSum[256];
    partialSum[threadIdx.x] = 0.0f;
    
    while (tid < n)
    {
        partialSum[threadIdx.x] += x[tid] * y[tid];
        tid += blockDim.x * gridDim.x;
    }
    
    __syncthreads();
    
    int i = blockDim.x/2;
    while (i != 0)
    {
        if (threadIdx.x < i)
            partialSum[threadIdx.x] += partialSum[threadIdx.x + i];
        
        __syncthreads();
        i /= 2;
    }
    
    if (threadIdx.x == 0)
        atomicAdd(result, partialSum[0]);
}

float chatblas_sdot(int n, float *x, float *y)
{
    float *d_x, *d_y, *d_result;
    float result = 0.0f;

    cudaMalloc((void **)&d_x, n * sizeof(float));
    cudaMalloc((void **)&d_y, n * sizeof(float));
    cudaMalloc((void **)&d_result, sizeof(float));

    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_result, 0.0f, sizeof(float));

    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    dotProduct<<<gridSize, blockSize>>>(n, d_x, d_y, d_result);

    cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_result);

    return result;
}
