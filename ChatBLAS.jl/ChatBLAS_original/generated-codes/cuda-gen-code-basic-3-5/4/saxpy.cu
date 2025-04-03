#include "chatblas_cuda.h"

__global__ void axpy_kernel(int n, float a, float *x, float *y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < n) {
        y[i] = a * x[i] + y[i];
    }
}

void chatblas_saxpy(int n, float a, float *x, float *y) {
    int block_size = 256;
    int num_blocks = (n + block_size - 1) / block_size;

    // Allocate device memory
    float *d_x, *d_y;
    cudaMalloc((void**)&d_x, n * sizeof(float));
    cudaMalloc((void**)&d_y, n * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, n * sizeof(float), cudaMemcpyHostToDevice);
    
    // Launch kernel
    axpy_kernel<<<num_blocks, block_size>>>(n, a, d_x, d_y);
    
    // Copy result back to host
    cudaMemcpy(y, d_y, n * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_x);
    cudaFree(d_y);
}
