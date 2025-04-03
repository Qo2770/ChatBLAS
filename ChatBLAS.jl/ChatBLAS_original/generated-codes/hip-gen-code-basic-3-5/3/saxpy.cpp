#include "chatblas_hip.h"

__global__ void saxpy_kernel(int n, float a, float *x, float *y) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        y[tid] += a * x[tid];
    }
}

void chatblas_saxpy(int n, float a, float *x, float *y) {
    float *d_x, *d_y;

    hipMalloc((void**)&d_x, n * sizeof(float));
    hipMalloc((void**)&d_y, n * sizeof(float));

    hipMemcpy(d_x, x, n * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_y, y, n * sizeof(float), hipMemcpyHostToDevice);

    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;

    saxpy_kernel<<<grid_size, block_size>>>(n, a, d_x, d_y);

    hipMemcpy(y, d_y, n * sizeof(float), hipMemcpyDeviceToHost);

    hipFree(d_x);
    hipFree(d_y);
}