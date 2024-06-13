#include "chatblas_hip.h"

__global__ void sswap_kernel(int n, float *x, float *y)
{
    int idx = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    if (idx < n){
        float temp = x[idx];
        x[idx] = y[idx];
        y[idx] = temp;
    }
}

void chatblas_sswap(int n, float *x, float *y)
{
    float *d_x, *d_y;
    hipMalloc((void**)&d_x, n * sizeof(float));
    hipMalloc((void**)&d_y, n * sizeof(float));

    hipMemcpy(d_x, x, n * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_y, y, n * sizeof(float), hipMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    hipLaunchKernelGGL(sswap_kernel, dim3(numBlocks), dim3(blockSize), 0, 0, n, d_x, d_y);

    hipMemcpy(x, d_x, n * sizeof(float), hipMemcpyDeviceToHost);
    hipMemcpy(y, d_y, n * sizeof(float), hipMemcpyDeviceToHost);

    hipFree(d_x);
    hipFree(d_y);
}
