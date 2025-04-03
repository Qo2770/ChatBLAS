#include "chatblas_cuda.h"

__global__ void computeNorm(float *x, float *result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        atomicAdd(result, x[idx] * x[idx]);
    }
}

float chatblas_snrm2(int n, float *x) {
    float *d_x, *d_result;
    float result = 0.0f;
    cudaMalloc(&d_x, n * sizeof(float));
    cudaMalloc(&d_result, sizeof(float));
    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
    
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    computeNorm<<<gridSize, blockSize>>>(d_x, d_result, n);
    
    cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_x);
    cudaFree(d_result);
    
    return sqrt(result);
}
