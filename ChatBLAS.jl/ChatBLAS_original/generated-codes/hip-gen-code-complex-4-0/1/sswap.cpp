#include "chatblas_hip.h"

__global__ void sswap_kernel(int n, float *x, float *y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        float temp = x[i];
        x[i] = y[i];
        y[i] = temp;
    }
} 

void chatblas_sswap(int n, float *x, float *y) {

    float *dx, *dy;

    int blk_size = 256;
    int grid_size = (n + blk_size - 1) / blk_size;

    hipMalloc((void **)&dx, n*sizeof(float));
    hipMalloc((void **)&dy, n*sizeof(float));

    hipMemcpy(dx, x, n*sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(dy, y, n*sizeof(float), hipMemcpyHostToDevice);

    hipLaunchKernelGGL(sswap_kernel, dim3(grid_size), dim3(blk_size), 0, 0, n, dx, dy);

    hipMemcpy(x, dx, n*sizeof(float), hipMemcpyDeviceToHost);
    hipMemcpy(y, dy, n*sizeof(float), hipMemcpyDeviceToHost);

    hipFree(dx);
    hipFree(dy);
}