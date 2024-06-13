#include "chatblas_hip.h"
#include <hip/hip_runtime.h>

__global__ void sdsdot_kernel(int n, float b, float* x, float* y, double* res) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    double sum = 0.0;
    if (index < n)
        sum = x[index] * y[index];
    atomicAdd(res, sum);
}

float chatblas_sdsdot(int n, float b, float* x, float* y) {
    float* d_x = nullptr;
    float* d_y = nullptr;
    double* d_res = nullptr;
    double* h_res = (double*) malloc(sizeof(double));
    *h_res = (double)b;

    hipMalloc((void**)&d_x, n*sizeof(float));
    hipMalloc((void**)&d_y, n*sizeof(float));
    hipMalloc((void**)&d_res, sizeof(double));

    hipMemcpy(d_x, x, n*sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_y, y, n*sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_res, h_res, sizeof(double), hipMemcpyHostToDevice);

    dim3 threads(1024);
    dim3 blocks((n + threads.x - 1) / threads.x);
    hipLaunchKernelGGL(sdsdot_kernel, blocks, threads, 0, 0, n, b, d_x, d_y, d_res);

    hipMemcpy(h_res, d_res, sizeof(double), hipMemcpyDeviceToHost);

    hipFree(d_x);
    hipFree(d_y);
    hipFree(d_res);

    float result = (float)*h_res;
    free(h_res);

    return result;
}
