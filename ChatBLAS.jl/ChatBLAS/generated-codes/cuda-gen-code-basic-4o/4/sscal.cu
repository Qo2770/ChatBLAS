#include "chatblas_cuda.h"
#include <cuda_runtime.h>
#include <stdio.h>

#define THREADS_PER_BLOCK 256

// CUDA kernel to scale a vector
__global__ void sscal_kernel(int n, float a, float *x) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        x[idx] *= a;
    }
}

// Host function to configure and launch the CUDA kernel
void chatblas_sscal(int n, float a, float *x) {
    float *d_x;
    size_t size = n * sizeof(float);

    // Allocate memory on the GPU
    cudaMalloc((void **)&d_x, size);

    // Copy the vector x to the GPU
    cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice);

    // Calculate the number of blocks needed
    int blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // Launch the kernel
    sscal_kernel<<<blocks, THREADS_PER_BLOCK>>>(n, a, d_x);

    // Check for any errors arising while launching
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch sscal_kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Wait for the GPU to finish
    cudaDeviceSynchronize();

    // Copy the result back to the host
    cudaMemcpy(x, d_x, size, cudaMemcpyDeviceToHost);

    // Free memory on the GPU
    cudaFree(d_x);
}
