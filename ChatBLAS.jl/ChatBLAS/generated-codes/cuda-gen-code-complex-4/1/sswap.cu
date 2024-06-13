#include "chatblas_cuda.h"

__global__ void sswap_kernel(int n, float *x, float *y) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        float a = x[idx];
        x[idx] = y[idx];
        y[idx] = a;
    }
}

void chatblas_sswap(int n, float *x, float *y) {
    float *d_x, *d_y;
    int size = n * sizeof(float);
    cudaMalloc((void **) &d_x, size);
    cudaMalloc((void **) &d_y, size);
    cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid =(n + threadsPerBlock - 1) / threadsPerBlock;
    sswap_kernel<<<blocksPerGrid, threadsPerBlock>>>(n, d_x, d_y);
    
    cudaMemcpy(x, d_x, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(y, d_y, size, cudaMemcpyDeviceToHost);
    cudaFree(d_x);
    cudaFree(d_y);
}