#include "chatblas_hip.h"

__global__ void sswap_kernel(int n, float *x, float *y) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n) {
        float temp = x[index];
        x[index] = y[index];
        y[index] = temp;
    }
}

void chatblas_sswap(int n, float *x, float *y) {
    const int blockSize = 256;
    const int gridSize = (n + blockSize - 1) / blockSize;

    float *x_gpu, *y_gpu;

    hipMalloc((void**)&x_gpu, n * sizeof(float));
    hipMalloc((void**)&y_gpu, n * sizeof(float));

    hipMemcpy(x_gpu, x, n * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(y_gpu, y, n * sizeof(float), hipMemcpyHostToDevice);

    sswap_kernel<<<gridSize, blockSize>>>(n, x_gpu, y_gpu);

    hipMemcpy(x, x_gpu, n * sizeof(float), hipMemcpyDeviceToHost);
    hipMemcpy(y, y_gpu, n * sizeof(float), hipMemcpyDeviceToHost);
  
    hipFree(x_gpu);
    hipFree(y_gpu);
}