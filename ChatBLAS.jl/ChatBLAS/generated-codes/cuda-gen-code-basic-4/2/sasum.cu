#include "chatblas_cuda.h"

__global__ void absolute_sum_kernel(int n, float *x, float *result)
{
    int idx = threadIdx.x;
    float sum = 0.0f;

    if(idx < n) 
        sum += fabs(x[idx]);

    atomicAdd(result, sum);
}

float chatblas_sasum(int n, float *x)
{
    float *result, *dev_result;
    float *dev_x;
    
    result = (float*)malloc(sizeof(float));
    *result = 0.0f;

    cudaMalloc((void**)&dev_result, sizeof(float));
    cudaMemcpy(dev_result, result, sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&dev_x, n * sizeof(float));
    cudaMemcpy(dev_x, x, n * sizeof(float), cudaMemcpyHostToDevice);

    absolute_sum_kernel<<<1, n>>>(n, dev_x, dev_result);

    cudaMemcpy(result, dev_result, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dev_result);
    cudaFree(dev_x);

    float final_result = *result;
    free(result);

    return final_result;
}
