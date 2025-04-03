#include "chatblas_cuda.h"

// Function to compute Euclidean norm of a vector x
float chatblas_snrm2(int n, float *x) {
    // Allocate memory for vector x on GPU
    float *d_x;
    cudaMalloc((void**)&d_x, n * sizeof(float));

    // Copy vector x from host to device
    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);

    // Initialize squared sum
    float sum_squared = 0.0f;

    // Compute squared sum of elements in vector x in parallel
    int threadsPerBlock = 256;
    int numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    squareSum<<<numBlocks, threadsPerBlock>>>(n, d_x, sum_squared);

    // Copy squared sum from the device to host
    float h_sum_squared;
    cudaMemcpy(&h_sum_squared, sum_squared, sizeof(float), cudaMemcpyDeviceToHost);

    // Free memory on the device
    cudaFree(d_x);

    // Compute and return the square root of the squared sum
    return sqrtf(h_sum_squared);
}

// Kernel function to compute squared sum of elements in vector x in parallel
__global__ void squareSum(int n, float *x, float *sum_squared) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    // Compute squared sum of elements in vector x
    if (index < n) {
        atomicAdd(sum_squared, x[index] * x[index]);
    }
}
