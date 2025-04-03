#include "chatblas_hip.h"

__global__ 
void saxpy_kernel(int n, float a, float *x, float *y) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        y[i] = a * x[i] + y[i];
    }
}

void chatblas_saxpy(int n, float a, float *x, float *y) {
    float *dx, *dy;

    hipMalloc(&dx, n*sizeof(float));
    hipMalloc(&dy, n*sizeof(float));

    hipMemcpy(dx, x, n*sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(dy, y, n*sizeof(float), hipMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    hipLaunchKernelGGL(saxpy_kernel, dim3(gridSize), dim3(blockSize), 0, 0, n, a, dx, dy);

    hipMemcpy(y, dy, n*sizeof(float), hipMemcpyDeviceToHost);

    hipFree(dx);
    hipFree(dy);
}
