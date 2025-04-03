#include "chatblas_cuda.h"

__global__ void sswap_kernel(int n, float *x, float *y) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < n){
        float temp = x[index];
        x[index] = y[index];
        y[index] = temp;
    }
}

void chatblas_sswap(int n, float *x, float *y) {
    float *d_x, *d_y;
    int size = n * sizeof(float);

    cudaMalloc((void **)&d_x, size);
    cudaMalloc((void **)&d_y, size);

    cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    sswap_kernel<<<blocksPerGrid, threadsPerBlock>>>(n, d_x, d_y);

    cudaMemcpy(x, d_x, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(y, d_y, size, cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_y);
}