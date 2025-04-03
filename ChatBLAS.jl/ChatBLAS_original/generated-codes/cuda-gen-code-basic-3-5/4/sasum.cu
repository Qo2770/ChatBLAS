#include "chatblas_cuda.h"

__global__ void sasum_kernel(int n, float *x, float *result) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
  
    float sum = 0.0f;
  
    for (int i = index; i < n; i += stride) {
        sum += fabsf(x[i]);
    }
  
    atomicAdd(result, sum);
}

float chatblas_sasum(int n, float *x) {
    float *d_x, *d_result;
    float result = 0.0f;
  
    // Allocate memory on the device
    cudaMalloc((void**)&d_x, n * sizeof(float));
    cudaMalloc((void**)&d_result, sizeof(float));
  
    // Copy input data from host to device
    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
  
    // Calculate grid and block sizes
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
  
    // Launch the kernel
    sasum_kernel<<<numBlocks, blockSize>>>(n, d_x, d_result);
  
    // Copy the result back to host
    cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
  
    // Free allocated memory on the device
    cudaFree(d_x);
    cudaFree(d_result);
  
    return result;
}
