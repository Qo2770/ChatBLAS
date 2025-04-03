#include "chatblas_hip.h"

__global__ void scopy_kernel( int n, float *x, float *y ) { 
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n) {
        y[i] = x[i];
    }
}

void chatblas_scopy(int n, float *x, float *y) {
    int thread_per_block = 256;
    int block_in_grid = (n + thread_per_block - 1) / thread_per_block;

    float *d_x, *d_y;

    hipMalloc((void **)&d_x, n * sizeof(float));
    hipMalloc((void **)&d_y, n * sizeof(float));

    hipMemcpy(d_x, x, n * sizeof(float), hipMemcpyHostToDevice);

    hipLaunchKernelGGL(scopy_kernel, dim3(block_in_grid), dim3(thread_per_block), 0, 0, n, d_x, d_y);

    hipMemcpy(y, d_y, n * sizeof(float), hipMemcpyDeviceToHost);

    hipFree(d_x);
    hipFree(d_y);
}