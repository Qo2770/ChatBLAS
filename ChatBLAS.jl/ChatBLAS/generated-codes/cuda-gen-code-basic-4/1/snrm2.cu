#include "chatblas_cuda.h"
#include <math.h>
#include <cublas_v2.h>

__global__ void euclidean_norm(const int n, const float* x, float* result) 
{
    __shared__ float cache[256];
    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    float temp = 0;
    
    while(index < n) 
    {
        temp += x[index] * x[index];
        index += blockDim.x * gridDim.x;
    }
    cache[tid] = temp;
    __syncthreads();
    
    int i = blockDim.x / 2;
    while(i != 0)
    {
        if(tid < i) 
        {
            cache[tid] += cache[tid + i];
        }
        __syncthreads();
        i /= 2;
    }
    
    if(tid == 0) 
    {
        result[blockIdx.x] = sqrtf(cache[0]);
    }
}

float chatblas_snrm2(int n, float *x) 
{
    float* d_x;
    float* d_result;
    
    cudaMalloc((void**)&d_x, n * sizeof(float));
    cudaMalloc((void**)&d_result, sizeof(float));
    
    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
    
    dim3 threadsPerBlock(256);
    dim3 numBlocks((n + threadsPerBlock.x - 1) / threadsPerBlock.x);
    euclidean_norm<<<numBlocks,threadsPerBlock>>>(n, d_x, d_result);
    
    float result;
    cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(d_x);
    cudaFree(d_result);
    
    return result;
}
