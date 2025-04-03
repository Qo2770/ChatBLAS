#include "chatblas_cuda.h"
#include <math.h>
#include <cuda_runtime.h>

__global__ void euclidean_norm(float *x, float *result, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    float sum = 0.0;
    for (int i = index; i < n; i += stride) {
        sum += x[i] * x[i];
    }
    
    atomicAdd(result, sum);
}

float chatblas_snrm2(int n, float *x) {
    float *d_x, *d_result;
    float result = 0.0;

    cudaMalloc((void **)&d_x, n * sizeof(float));
    cudaMalloc((void **)&d_result, sizeof(float));

    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_result, 0, sizeof(float));

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    euclidean_norm<<<numBlocks, blockSize>>>(d_x, d_result, n);

    cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_result);

    return sqrt(result);
}