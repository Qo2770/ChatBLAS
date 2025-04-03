#include "chatblas_hip.h"

__global__ void snrm2_kernel(int n, float *x, float *res)
{
    const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    float temp = 0.0;
    if(idx < n)
    {
        temp = x[idx];
        atomicAdd(res,(temp * temp));
    }
}

float chatblas_snrm2(int n, float *x)
{
    const int blockSize = 256;
    const int gridSize = (n + blockSize - 1) / blockSize;
    float *d_x, *d_res, *h_res;
    h_res = (float*)calloc(1, sizeof(float));
    cudaMalloc((void**)&d_x, n * sizeof(float));
    cudaMalloc((void**)&d_res, sizeof(float));
    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_res, 0, sizeof(float));
    snrm2_kernel<<<gridSize, blockSize>>>(n, d_x, d_res);
    cudaMemcpy(h_res, d_res, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_x);
    cudaFree(d_res);
    float norm = sqrt(*h_res);
    free(h_res);
    return norm;
}