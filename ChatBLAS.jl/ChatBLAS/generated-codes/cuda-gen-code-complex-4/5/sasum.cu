#include "chatblas_cuda.h"

__global__ void sasum_kernel(int n, float *x, float *sum)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) 
    {
        sum[i]=fabs(x[i]);
    }
}

float chatblas_sasum(int n, float *x)
{
    float *d_x, *d_sum, *partial_sum;
    float sum = 0.0f;
    int blockSize=256;
    int grid=(n+blockSize-1)/blockSize;
  
    cudaMalloc((void**)&d_x, n*sizeof(float));
    cudaMalloc((void**)&d_sum, n*sizeof(float));
  
    cudaMemcpy(d_x, x, n*sizeof(float), cudaMemcpyHostToDevice);
  
    sasum_kernel<<<grid, blockSize>>>(n, d_x, d_sum);
  
    partial_sum = (float*) malloc(n*sizeof(float));   
    cudaMemcpy(partial_sum, d_sum, n*sizeof(float), cudaMemcpyDeviceToHost);
  
    for(int i=0; i<n; i++)
    {
        sum += partial_sum[i];
    }
  
    free(partial_sum);
    cudaFree(d_x);
    cudaFree(d_sum);
  
    return sum;
}