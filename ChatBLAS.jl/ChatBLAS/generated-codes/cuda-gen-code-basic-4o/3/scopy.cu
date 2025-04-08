#include "chatblas_cuda.h"
#include <cuda_runtime.h>

__global__ void vectorCopyKernel(int n, const float *x, float *y) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = x[idx];
    }
}

void chatblas_scopy(int n, float *x, float *y) {
    // Define device pointers
    float *d_x, *d_y;

    // Allocate memory on the device
    cudaMalloc((void**)&d_x, n * sizeof(float));
    cudaMalloc((void**)&d_y, n * sizeof(float));

    // Copy input vector from host to device
    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);

    // Define block and grid sizes
    int blockSize = 256; // You can choose a suitable block size based on your hardware
    int gridSize = (n + blockSize - 1) / blockSize;

    // Launch the kernel
    vectorCopyKernel<<<gridSize, blockSize>>>(n, d_x, d_y);

    // Copy result back to host
    cudaMemcpy(y, d_y, n * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_x);
    cudaFree(d_y);
}
