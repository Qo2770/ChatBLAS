#include "chatblas_cuda.h"

__global__ void dotProduct(int n, float *x, float *y, float *result) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    
    float localSum = 0.0f;
    for (int i = index; i < n; i += stride) {
        localSum += x[i] * y[i];
    }
    
    atomicAdd(result, localSum);
}

float chatblas_sdot(int n, float *x, float *y) {
    float *dev_x, *dev_y, *dev_result;
    
    // Allocate memory on the device
    cudaMalloc((void**)&dev_x, n * sizeof(float));
    cudaMalloc((void**)&dev_y, n * sizeof(float));
    cudaMalloc((void**)&dev_result, sizeof(float));
    
    // Copy input vectors from host to device memory
    cudaMemcpy(dev_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_y, y, n * sizeof(float), cudaMemcpyHostToDevice);
    
    // Set initial value of the result on the device to 0
    cudaMemset(dev_result, 0, sizeof(float));
    
    // Launch kernel to compute the dot product in parallel
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    dotProduct<<<numBlocks, blockSize>>>(n, dev_x, dev_y, dev_result);
    
    // Copy the result back from the device to the host
    float result;
    cudaMemcpy(&result, dev_result, sizeof(float), cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(dev_x);
    cudaFree(dev_y);
    cudaFree(dev_result);
    
    return result;
}
