#include "chatblas_cuda.h"
#include <cuda_runtime.h>

// CUDA kernel to scale a vector
__global__ void scaleKernel(int n, float a, float *x) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        x[idx] *= a;
    }
}

// Host function to scale a vector using CUDA
void chatblas_sscal(int n, float a, float *x) {
    if (n <= 0) return;

    // Choose a reasonable number of threads per block
    int threadsPerBlock = 256;
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock; // Calculate number of blocks needed

    // Allocate memory on the GPU
    float *d_x;
    size_t size = n * sizeof(float);
    cudaMalloc((void **)&d_x, size);

    // Copy vector from host to GPU
    cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice);

    // Launch kernel
    scaleKernel<<<blocks, threadsPerBlock>>>(n, a, d_x);

    // Copy result back to host
    cudaMemcpy(x, d_x, size, cudaMemcpyDeviceToHost);

    // Free allocated memory on the GPU
    cudaFree(d_x);
}
