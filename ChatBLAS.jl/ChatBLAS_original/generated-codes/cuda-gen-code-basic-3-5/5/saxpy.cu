#include "chatblas_cuda.h"

__global__
void saxpy(int n, float a, float *x, float *y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = a * x[i] + y[i];
    }
}

void chatblas_saxpy(int n, float a, float *x, float *y) {
    float *d_x, *d_y;

    // Allocate device memory
    cudaMalloc((void **)&d_x, n * sizeof(float));
    cudaMalloc((void **)&d_y, n * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, n * sizeof(float), cudaMemcpyHostToDevice);

    // Define grid and block sizes for CUDA kernel execution
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    // Launch the CUDA kernel
    saxpy<<<gridSize, blockSize>>>(n, a, d_x, d_y);

    // Copy the result back from device to host
    cudaMemcpy(y, d_y, n * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_x);
    cudaFree(d_y);
}
