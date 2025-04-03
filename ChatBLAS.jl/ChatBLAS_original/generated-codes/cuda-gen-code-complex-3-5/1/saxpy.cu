#include "chatblas_cuda.h"

__global__ void saxpy_kernel(int n, float a, float *x, float *y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = a * x[i] + y[i];
    }
}

void chatblas_saxpy(int n, float a, float *x, float *y) {
    float *d_x, *d_y;
    int size = n * sizeof(float);

    // Allocate GPU memory for vectors
    cudaMalloc((void **)&d_x, size);
    cudaMalloc((void **)&d_y, size);

    // Transfer data from CPU to GPU
    cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, size, cudaMemcpyHostToDevice);

    // Define block and grid size
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    // Call kernel function
    saxpy_kernel<<<blocksPerGrid, threadsPerBlock>>>(n, a, d_x, d_y);

    // Transfer data from GPU to CPU
    cudaMemcpy(y, d_y, size, cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_x);
    cudaFree(d_y);
}