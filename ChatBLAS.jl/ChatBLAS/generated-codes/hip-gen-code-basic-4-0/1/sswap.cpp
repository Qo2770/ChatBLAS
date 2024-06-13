#include "hip/hip_runtime.h"

__global__ void sswap_kernel(int n, float *x, float *y) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        float temp = x[idx];
        x[idx] = y[idx];
        y[idx] = temp;
    }
}

void chatblas_sswap(int n, float *x, float *y) {
    float *dp_x;
    float *dp_y;
    hipMalloc((void **)&dp_x, n * sizeof(float));
    hipMalloc((void **)&dp_y, n * sizeof(float));

    hipMemcpy(dp_x, x, n * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(dp_y, y, n * sizeof(float), hipMemcpyHostToDevice);

    int threadNumber = 256;
    int blockSize = (n + threadNumber - 1) / threadNumber;
    hipLaunchKernelGGL(sswap_kernel, dim3(blockSize), dim3(threadNumber), 0, 0, n, dp_x, dp_y);

    hipMemcpy(x, dp_x, n * sizeof(float), hipMemcpyDeviceToHost);
    hipMemcpy(y, dp_y, n * sizeof(float), hipMemcpyDeviceToHost);

    hipFree(dp_x);
    hipFree(dp_y);
}
