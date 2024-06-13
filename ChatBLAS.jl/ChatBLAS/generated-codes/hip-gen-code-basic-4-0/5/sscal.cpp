#include "chatblas_hip.h"

__global__ void sscal_kernel( int n, float a , float *x ) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n)
        x[idx] = a * x[idx];
}

void chatblas_sscal( int n, float a, float *x) {
    float *device_x;
    hipMalloc(&device_x, n * sizeof(float));

    hipMemcpy(device_x, x, n * sizeof(float), hipMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    hipLaunchKernelGGL(sscal_kernel, dim3(numBlocks), dim3(blockSize), 0, 0, n, a, device_x);

    hipMemcpy(x, device_x, n * sizeof(float), hipMemcpyDeviceToHost);

    hipFree(device_x);
}
