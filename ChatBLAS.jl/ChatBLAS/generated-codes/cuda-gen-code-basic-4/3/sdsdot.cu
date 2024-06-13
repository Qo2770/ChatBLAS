#include "chatblas_cuda.h"

// CUDA Kernel Function
__global__ void dot_product(int n, float b, float *x, float *y, float *result) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        result[tid] = ((double)x[tid] * (double)y[tid]) + (double)b;
    }
}

// Main Function
float chatblas_sdsdot(int n, float b, float *x, float *y) {
    float *d_x, *d_y, *d_result;

    // Allocate memory on the device
    cudaMalloc((void**)&d_x, n * sizeof(float));
    cudaMalloc((void**)&d_y, n * sizeof(float));
    cudaMalloc((void**)&d_result, n * sizeof(float));

    // Copy vectors from host to device
    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, n * sizeof(float), cudaMemcpyHostToDevice);
    
    // Set grid and block dimensions
    dim3 blockDim(256);
    dim3 gridDim((n + blockDim.x - 1) / blockDim.x);
    
    // Launch CUDA kernel
    dot_product<<<gridDim, blockDim>>>(n, b, d_x, d_y, d_result);
    
    float *h_result = (float*) malloc(n * sizeof(float));
    // Copy the result from device to host
    cudaMemcpy(h_result, d_result, n * sizeof(float), cudaMemcpyDeviceToHost);

    // Sum the results
    float sum = 0;
    for (int i = 0; i < n; ++i) {
        sum += h_result[i];
    }

    // Free the device memory
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_result);

    // Free the host memory
    free(h_result);

    return sum;
}
