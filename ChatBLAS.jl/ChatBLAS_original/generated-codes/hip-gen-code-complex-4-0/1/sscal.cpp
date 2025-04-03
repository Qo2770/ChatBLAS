#include "chatblas_hip.h"

__global__ void sscal_kernel( int n, float a , float *x ) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        x[idx] = a * x[idx];
    }
}

void chatblas_sscal( int n, float a, float *x) {
    float *d_x;
    hipMalloc(&d_x, n * sizeof(float));
    hipMemcpy(d_x, x, n * sizeof(float), hipMemcpyHostToDevice);

    dim3 dimBlock(256);
    dim3 dimGrid((n + dimBlock.x - 1) / dimBlock.x);
    hipLaunchKernelGGL(sscal_kernel, dimGrid, dimBlock, 0, 0, n, a, d_x);

    hipMemcpy(x, d_x, n * sizeof(float), hipMemcpyDeviceToHost);
    hipFree(d_x);
}