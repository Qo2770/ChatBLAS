#include "chatblas_cuda.h"

// The kernel function to swap two vectors.
__global__ void chatblas_sswap_kernel(int n, float *x, float *y) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Make sure we don't go out of bounds
    if (tid < n) {
        // Perform the swap
        float tmp = x[tid];
        x[tid] = y[tid];
        y[tid] = tmp;
    }
}

// The chatblas_sswap function
void chatblas_sswap(int n, float *x, float *y) {
    // Number of threads in each thread block
    int threadsPerBlock = 256;

    // Number of thread blocks in grid
    int blocksPerGrid =(n + threadsPerBlock - 1) / threadsPerBlock;

    // Launch a kernel on the GPU with one thread for each element.
    chatblas_sswap_kernel<<<blocksPerGrid, threadsPerBlock>>>(n, x, y);

    // Error checking
    cudaError err = cudaGetLastError();
    if (err != cudaSuccess) 
        printf("Error: %s\n", cudaGetErrorString(err));
}
