#include "chatblas_cuda.h"

__global__ void dotProductKernel(int n, float *x, float *y, float *partialResults) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        partialResults[idx] = x[idx] * y[idx];
    }
}

float chatblas_sdot(int n, float *x, float *y) {
    float *d_x, *d_y, *d_partialResults;
    float *h_partialResults = (float *)malloc(n * sizeof(float));
    float result = 0.0f;

    cudaMalloc((void **)&d_x, n * sizeof(float));
    cudaMalloc((void **)&d_y, n * sizeof(float));
    cudaMalloc((void **)&d_partialResults, n * sizeof(float));

    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, n * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    dotProductKernel<<<gridSize, blockSize>>>(n, d_x, d_y, d_partialResults);

    cudaMemcpy(h_partialResults, d_partialResults, n * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; i++) {
        result += h_partialResults[i];
    }

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_partialResults);
    free(h_partialResults);

    return result;
}
