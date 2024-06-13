#include "chatblas_cuda.h"

__global__ void kernel_scal(int n, float a, float *x) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        x[i] = a * x[i];
    }
}

void chatblas_sscal( int n, float a , float *x) {
    float *dx;

    // Allocate device memory and copy vector from host to device
    cudaMalloc((void **)&dx, n * sizeof(float));
    cudaMemcpy(dx, x, n * sizeof(float), cudaMemcpyHostToDevice);

    // Calculate grid and block sizes
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    // Call the kernel
    kernel_scal<<<gridSize, blockSize>>>(n, a, dx);

    // Copy result vector from device to host
    cudaMemcpy(x, dx, n * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(dx);
}
