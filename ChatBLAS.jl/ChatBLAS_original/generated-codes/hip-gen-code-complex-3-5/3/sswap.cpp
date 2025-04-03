#include "chatblas_hip.h"

__global__ void sswap_kernel(int n, float *x, float *y) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        float temp = x[index];
        x[index] = y[index];
        y[index] = temp;
    }
}

void chatblas_sswap(int n, float *x, float *y) {
    float *d_x, *d_y;
    hipMalloc(&d_x, n * sizeof(float));
    hipMalloc(&d_y, n * sizeof(float));
    
    hipMemcpy(d_x, x, n * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_y, y, n * sizeof(float), hipMemcpyHostToDevice);
    
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    
    sswap_kernel<<<gridSize, blockSize>>>(n, d_x, d_y);
    
    hipMemcpy(x, d_x, n * sizeof(float), hipMemcpyDeviceToHost);
    hipMemcpy(y, d_y, n * sizeof(float), hipMemcpyDeviceToHost);
    
    hipFree(d_x);
    hipFree(d_y);
}