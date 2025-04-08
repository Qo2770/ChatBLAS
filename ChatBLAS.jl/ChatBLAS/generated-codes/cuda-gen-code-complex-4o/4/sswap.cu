#include "chatblas_cuda.h"

__global__ void sswap_kernel(int n, float *x, float *y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float temp = x[i];
        x[i] = y[i];
        y[i] = temp;
    }
}

void chatblas_sswap(int n, float *x, float *y) {
    float *d_x, *d_y;
    size_t size = n * sizeof(float);

    cudaMalloc((void**)&d_x, size);
    cudaMalloc((void**)&d_y, size);

    cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, size, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    sswap_kernel<<<numBlocks, blockSize>>>(n, d_x, d_y);

    cudaMemcpy(x, d_x, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(y, d_y, size, cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_y);
}