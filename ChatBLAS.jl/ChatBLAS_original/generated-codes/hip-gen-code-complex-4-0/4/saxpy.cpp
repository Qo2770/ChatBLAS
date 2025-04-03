#include "chatblas_hip.h"

__global__ void saxpy_kernel(int n, float a, float *x, float *y) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n)
        y[i] = a * x[i] + y[i];
}

void chatblas_saxpy(int n, float a, float *x, float *y) {
    float *d_x, *d_y;
    
    cudaMalloc((void **)&d_x, n * sizeof(float));
    cudaMalloc((void **)&d_y, n * sizeof(float));
    
    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, n * sizeof(float), cudaMemcpyHostToDevice);
    
    int threadsPerBlock = 256;
    int blocksPerGrid =(n + threadsPerBlock - 1) / threadsPerBlock;
    saxpy_kernel<<<blocksPerGrid, threadsPerBlock>>>(n, a, d_x, d_y);

    cudaMemcpy(y, d_y, n * sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(d_x);
    cudaFree(d_y);
}