#include "chatblas_cuda.h"
#include <math.h>

// CUDA kernel to calculate the squares of each element
__global__ void vecSquares(float *x, int n) {
    // Get the global thread ID
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    // Boundary check
    if (index < n) 
        x[index] = x[index] * x[index];
}

float chatblas_snrm2(int n, float *x) {
    float *d_x;
    float sum = 0.0f;

    // Allocate device memory
    cudaMalloc((void**)&d_x, n * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);

    // Calculate the number of blocks of threads
    int blockSize = 1024;
    int gridSize = (int)ceil((float)n/blockSize);

    // Launch the CUDA Kernel
    vecSquares<<<gridSize, blockSize>>>(d_x, n);

    // Wait for CUDA to finish
    cudaDeviceSynchronize();

    // Copy data back from device to host
    cudaMemcpy(x, d_x, n * sizeof(float), cudaMemcpyDeviceToHost);

    // Deallocate device memory
    cudaFree(d_x);

    //Sum of squares for calculating Euclidean Norm
    for(int i = 0; i < n; i++) {
        sum += x[i];
    }

    //Taking square root of sum to calculate Euclidean Norm
    return sqrt(sum);
}
