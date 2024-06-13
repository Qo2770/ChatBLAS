#include "chatblas_hip.h"

__global__ void scopy_kernel( int n, float *x, float *y ) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        y[index] = x[index];
    }
}

void chatblas_scopy(int n, float *x, float *y) {
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    float *d_x, *d_y;
    hipMalloc(&d_x, n * sizeof(float));
    hipMalloc(&d_y, n * sizeof(float);

    hipMemcpy(d_x, x, n * sizeof(float), hipMemcpyHostToDevice);

    scopy_kernel<<<numBlocks, blockSize>>>(n, d_x, d_y);

    hipMemcpy(y, d_y, n * sizeof(float), hipMemcpyDeviceToHost);

    hipFree(d_x);
    hipFree(d_y);
}