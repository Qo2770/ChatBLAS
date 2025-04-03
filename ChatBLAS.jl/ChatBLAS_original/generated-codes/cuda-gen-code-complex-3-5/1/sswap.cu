#include "chatblas_cuda.h"

__global__ void sswap_kernel(int n, float *x, float *y) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id < n) {
        float temp = x[id];
        x[id] = y[id];
        y[id] = temp;
    }
}

void chatblas_sswap(int n, float *x, float *y) {
    // Allocate GPU memory for x and y
    float *dev_x, *dev_y;
    cudaMalloc((void**)&dev_x, n * sizeof(float));
    cudaMalloc((void**)&dev_y, n * sizeof(float));
    
    // Copy data from CPU to GPU
    cudaMemcpy(dev_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_y, y, n * sizeof(float), cudaMemcpyHostToDevice);
    
    // Define block size and number of blocks
    int block_size = 256;
    int num_blocks = (n + block_size - 1) / block_size;
    
    // Call the kernel
    sswap_kernel<<<num_blocks, block_size>>>(n, dev_x, dev_y);
    
    // Copy result from GPU to CPU
    cudaMemcpy(x, dev_x, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(y, dev_y, n * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Free GPU memory
    cudaFree(dev_x);
    cudaFree(dev_y);
}
