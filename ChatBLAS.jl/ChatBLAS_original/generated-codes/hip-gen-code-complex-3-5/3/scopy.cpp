#include "chatblas_hip.h"

__global__ void scopy_kernel( int n, float *x, float *y ) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        y[tid] = x[tid];
    }
}

void chatblas_scopy(int n, float *x, float *y) {
    float *d_x, *d_y;
    hipMalloc(&d_x, n * sizeof(float));
    hipMalloc(&d_y, n * sizeof(float));

    hipMemcpy(d_x, x, n * sizeof(float), hipMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    scopy_kernel<<<numBlocks, blockSize>>>(n, d_x, d_y);

    hipMemcpy(y, d_y, n * sizeof(float), hipMemcpyDeviceToHost);

    hipFree(d_x);
    hipFree(d_y);
}