#include <cuda_runtime.h>
#include "chatblas_cuda.h"

// CUDA kernel to scale the vector
__global__ void scaleKernel(int n, float a, float *x) {
    // Calculate the global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Ensure the index is within bounds
    if (idx < n) {
        x[idx] *= a;
    }
}

void chatblas_sscal(int n, float a, float *x) {
    // Assume x is already allocated on the GPU
    // Define the number of threads per block
    int threadsPerBlock = 256;

    // Calculate the number of blocks needed
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the scaling kernel
    scaleKernel<<<blocksPerGrid, threadsPerBlock>>>(n, a, x);

    // Optional: Synchronize to wait for kernel completion
    cudaDeviceSynchronize();

    // Optional: Check for any CUDA errors
    #ifdef DEBUG
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA Error: %s\n", cudaGetErrorString(err));
        }
    #endif
}

