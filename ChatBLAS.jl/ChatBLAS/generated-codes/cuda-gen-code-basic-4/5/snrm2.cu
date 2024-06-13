#include "chatblas_cuda.h"
#include <math.h>

__global__ void vectorNormKernel(int n, float *d_x, float *d_result)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx < n)
    {
        float element = d_x[idx];
        atomicAdd(d_result, element * element);
    }
}

float chatblas_snrm2(int n, float *x)
{
    float *d_x, *d_result;
    float result = 0;

    cudaMalloc((void **) &d_x, n*sizeof(float));
    cudaMemcpy(d_x, x, n*sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc((void **) &d_result, sizeof(float));
    cudaMemcpy(d_result, &result, sizeof(float), cudaMemcpyHostToDevice);

    vectorNormKernel<<<(n+255)/256, 256>>>(n, d_x, d_result);
    
    cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_result);

    return sqrt(result);
}
