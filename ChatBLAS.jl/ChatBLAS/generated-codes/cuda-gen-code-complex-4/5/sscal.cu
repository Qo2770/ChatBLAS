#include "chatblas_cuda.h"

__global__ void sscal_kernel(int n, float a, float *x) {
    int thread_idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (thread_idx < n) {
        x[thread_idx] = a * x[thread_idx];
    }
}

void chatblas_sscal(int n, float a, float *x) {
    float *device_x;
    int size = n * sizeof(float);

    cudaMalloc((void **) &device_x, size);

    cudaMemcpy(device_x, x, size, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    
    sscal_kernel<<<numBlocks, blockSize>>>(n, a, device_x);

    cudaMemcpy(x, device_x, size, cudaMemcpyDeviceToHost);

    cudaFree(device_x);
}