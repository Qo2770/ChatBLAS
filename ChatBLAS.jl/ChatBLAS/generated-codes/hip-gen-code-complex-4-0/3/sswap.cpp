#include "chatblas_hip.h"
#include "hip/hip_runtime.h"

__global__ void sswap_kernel(int n, float *x, float *y) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n) {
        float temp = x[index];
        x[index] = y[index];
        y[index] = temp;
    }
}

void chatblas_sswap(int n, float *x, float *y) {
    float *dx, *dy;
    
    hipMalloc((void**)&dx, n * sizeof(float));
    hipMalloc((void**)&dy, n * sizeof(float));

    hipMemcpy(dx, x, n * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(dy, y, n * sizeof(float), hipMemcpyHostToDevice);
     
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
     
    hipLaunchKernelGGL(sswap_kernel, dim3(gridSize), dim3(blockSize), 0, 0, n, dx, dy);

    hipMemcpy(x, dx, n * sizeof(float), hipMemcpyDeviceToHost);
    hipMemcpy(y, dy, n * sizeof(float), hipMemcpyDeviceToHost);
     
    hipFree(dx);
    hipFree(dy);
}