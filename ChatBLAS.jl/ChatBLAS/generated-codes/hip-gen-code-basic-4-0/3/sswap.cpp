#include "chatblas_hip.h"
#include "hip/hip_runtime.h"


__global__ void sswap_kernel(int n, float *x, float *y)
{
    int i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if (i < n)
    {
        float temp = x[i];
        x[i] = y[i];
        y[i] = temp;
    }
}

void chatblas_sswap(int n, float *x, float *y)
{
    float *d_x, *d_y;

    // allocate device memory
    hipMalloc((void**)&d_x, n*sizeof(float));
    hipMalloc((void**)&d_y, n*sizeof(float));

    // copy vectors from host to device
    hipMemcpy(d_x, x, n*sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_y, y, n*sizeof(float), hipMemcpyHostToDevice);

    // determine grid and block size
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    // call kernel
    hipLaunchKernelGGL(sswap_kernel, dim3(gridSize), dim3(blockSize), 0, 0, n, d_x, d_y);

    // copy results from device to host
    hipMemcpy(x, d_x, n*sizeof(float), hipMemcpyDeviceToHost);
    hipMemcpy(y, d_y, n*sizeof(float), hipMemcpyDeviceToHost);

    // free device memory
    hipFree(d_x);
    hipFree(d_y);
}
