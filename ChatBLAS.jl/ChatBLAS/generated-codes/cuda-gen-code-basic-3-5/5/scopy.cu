#include "chatblas_cuda.h"

void chatblas_scopy(int n, float *x, float *y) {
    // Allocate memory on the GPU
    float *d_x, *d_y;
    cudaMalloc((void **)&d_x, sizeof(float) * n);
    cudaMalloc((void **)&d_y, sizeof(float) * n);

    // Copy input vectors from host memory to GPU buffers
    cudaMemcpy(d_x, x, sizeof(float) * n, cudaMemcpyHostToDevice);

    // Launch the copy kernel
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    copyKernel<<<gridSize, blockSize>>>(n, d_x, d_y);

    // Copy output vector from GPU buffer to host memory
    cudaMemcpy(y, d_y, sizeof(float) * n, cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_x);
    cudaFree(d_y);
}

__global__ void copyKernel(int n, float *x, float *y) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n)
        y[index] = x[index];
}
