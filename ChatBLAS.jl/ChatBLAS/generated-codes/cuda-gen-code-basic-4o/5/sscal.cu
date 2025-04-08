#include "chatblas_cuda.h"
#include <cuda_runtime.h>
#include <stdio.h>

// CUDA kernel to scale a vector by a constant
__global__ void scaleVectorKernel(int n, float a, float *x) {
    // Calculate the global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Ensure we don't access out of bounds
    if (idx < n) {
        x[idx] *= a;
    }
}

// Host function to scale a vector by a constant
void chatblas_sscal(int n, float a, float *x) {
    // Error variables
    cudaError_t err;

    // Allocate memory on the device
    float *d_x;
    err = cudaMalloc((void **)&d_x, n * sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device vector x (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy vector x from host memory to device memory
    err = cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy vector x from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Define block size and grid size
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the CUDA kernel
    scaleVectorKernel<<<blocksPerGrid, threadsPerBlock>>>(n, a, d_x);

    // Check for kernel launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch scaleVectorKernel kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the result vector from device memory to host memory
    err = cudaMemcpy(x, d_x, n * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy vector x from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Free device memory
    err = cudaFree(d_x);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to free device vector x (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}
