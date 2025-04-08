#include "chatblas_cuda.h"

__global__ void swap_vectors_kernel(int n, float *x, float *y) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float temp = x[idx];
        x[idx] = y[idx];
        y[idx] = temp;
    }
}

void chatblas_sswap(int n, float *x, float *y) {
    // Define the number of threads per block
    int threadsPerBlock = 256;
    // Calculate the number of blocks needed
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    // Allocate device memory
    float *d_x, *d_y;
    cudaMalloc((void**)&d_x, n * sizeof(float));
    cudaMalloc((void**)&d_y, n * sizeof(float));

    // Copy the input vectors from host to device
    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, n * sizeof(float), cudaMemcpyHostToDevice);

    // Launch the kernel to swap the vectors
    swap_vectors_kernel<<<blocksPerGrid, threadsPerBlock>>>(n, d_x, d_y);

    // Copy the swapped vectors from device back to host
    cudaMemcpy(x, d_x, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(y, d_y, n * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_x);
    cudaFree(d_y);
}
