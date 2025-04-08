#include "chatblas_cuda.h"
#include <stdio.h>
#include <cuda_runtime.h>

// CUDA kernel to scale a vector
__global__ void scale_vector(int n, float a, float *x) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        x[i] *= a;
    }
}

void chatblas_sscal(int n, float a, float *x) {
    // Define block size for CUDA kernel
    const int blockSize = 256;
    const int numBlocks = (n + blockSize - 1) / blockSize;

    // Allocate memory on the GPU
    float *d_x;
    cudaMalloc((void **)&d_x, n * sizeof(float));

    // Copy the input vector from host to device
    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);

    // Launch the CUDA kernel
    scale_vector<<<numBlocks, blockSize>>>(n, a, d_x);

    // Copy the result back to the host
    cudaMemcpy(x, d_x, n * sizeof(float), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_x);
}
