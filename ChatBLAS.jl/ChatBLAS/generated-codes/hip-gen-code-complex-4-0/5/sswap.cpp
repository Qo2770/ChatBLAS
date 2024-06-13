#include "chatblas_hip.h"

__global__ void sswap_kernel(int n, float *x, float *y) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n) {
        float tmp = x[index];
        x[index] = y[index];
        y[index] = tmp;
    }
}

void chatblas_sswap(int n, float *x, float *y) {
    float *dev_x, *dev_y;
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    hipMalloc((void**)&dev_x, n * sizeof(float));
    hipMalloc((void**)&dev_y, n * sizeof(float));

    hipMemcpy(dev_x, x, n * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(dev_y, y, n * sizeof(float), hipMemcpyHostToDevice);

    hipLaunchKernelGGL(sswap_kernel, dim3(numBlocks), dim3(blockSize), 0, 0, n, dev_x, dev_y);

    hipMemcpy(x, dev_x, n * sizeof(float), hipMemcpyDeviceToHost);
    hipMemcpy(y, dev_y, n * sizeof(float), hipMemcpyDeviceToHost);

    hipFree(dev_x);
    hipFree(dev_y);
}