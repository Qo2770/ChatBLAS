#include "chatblas_cuda.h"

__global__ void snrm2_kernel(int n, float *x, float *res) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float element = 0.0;

    if(idx < n) {
        element = x[idx];
        res[idx] = element * element;
    }
}

float chatblas_snrm2(int n, float *x) {
    int blockDim = 256;
    int gridDim = (n + blockDim - 1) / blockDim;
    
    float *x_dev = NULL;
    float *res_dev = NULL;
    float *res = (float*) malloc(n * sizeof(float));
    float result = 0.0;

    cudaMalloc((void**)&x_dev, n * sizeof(float));
    cudaMalloc((void**)&res_dev, n * sizeof(float));
    
    cudaMemcpy(x_dev, x, n * sizeof(float), cudaMemcpyHostToDevice);

    snrm2_kernel<<<gridDim, blockDim>>>(n, x_dev, res_dev);
    
    cudaMemcpy(res, res_dev, n * sizeof(float), cudaMemcpyDeviceToHost);
    
    for(int i = 0; i < n; i++) {
        result += res[i];
    }
    result = sqrt(result);

    cudaFree(res_dev);
    cudaFree(x_dev);
    free(res);

    return result;
}