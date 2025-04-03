#include "chatblas_cuda.h"

__global__ void abs_sum_kernel(float* d_x, float* d_out, int n) 
{
    int index  = threadIdx.x + blockDim.x * blockIdx.x;
    int stride = blockDim.x * gridDim.x;
    __shared__ float cache[256];
  
    float temp = 0.0;
    while(index < n)
    {
        temp += fabsf(d_x[index]);
        index += stride;
    }
    cache[threadIdx.x] = temp;
  
    __syncthreads();

    if(threadIdx.x == 0)
    {
        float sum_val = 0.0;
        for(int i=0; i<blockDim.x; i++) 
        {
            sum_val += cache[i];
        }
        atomicAdd(d_out, sum_val);
    }
}

float chatblas_sasum(int n, float *x)
{
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    float *d_x, *d_out;
    
    float result = 0.0;
    float *h_out = &result;

    cudaMalloc(&d_x, n*sizeof(float));
    cudaMalloc((void**) &d_out, sizeof(float));
   
    cudaMemcpy(d_x, x, n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_out, h_out, sizeof(float), cudaMemcpyHostToDevice);

    abs_sum_kernel<<<numBlocks, blockSize>>>(d_x, d_out, n);

    cudaMemcpy(h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_out);

    return result;
}
