#include "chatblas_hip.h"

__global__ void saxpy_kernel(int n, float a, float *x, float *y) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) y[i] = a*x[i] + y[i];
}

void chatblas_saxpy(int n, float a, float *x, float *y) {
    float *d_x, *d_y;
    size_t size = n*sizeof(float);

    hipMalloc((void **)&d_x, size);
    hipMalloc((void **)&d_y, size);

    hipMemcpy(d_x, x, size, hipMemcpyHostToDevice);
    hipMemcpy(d_y, y, size, hipMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    hipLaunchKernelGGL(saxpy_kernel, dim3(numBlocks), dim3(blockSize), 0, 0, n, a, d_x, d_y);

    hipMemcpy(y, d_y, size, hipMemcpyDeviceToHost);

    hipFree(d_x);
    hipFree(d_y);
}