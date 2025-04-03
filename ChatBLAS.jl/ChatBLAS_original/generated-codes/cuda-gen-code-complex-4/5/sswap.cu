#include "chatblas_cuda.h"

__global__ void sswap_kernel(int n, float *x, float *y) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < n) {
        float tmp = x[i];
        x[i] = y[i];
        y[i] = tmp;
    }
}

void chatblas_sswap(int n, float *x, float *y) {
    float *dx, *dy;

    cudaMalloc(&dx, n*sizeof(float));
    cudaMemcpy(dx, x, n*sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&dy, n*sizeof(float));
    cudaMemcpy(dy, y, n*sizeof(float), cudaMemcpyHostToDevice);

    int blocks = (n + 255) / 256;
    int threads = (n < 256) ? n : 256;

    sswap_kernel<<<blocks, threads>>>(n, dx, dy);
        
    cudaMemcpy(x, dx, n*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(y, dy, n*sizeof(float), cudaMemcpyDeviceToHost);
        
    cudaFree(dx);
    cudaFree(dy);
}