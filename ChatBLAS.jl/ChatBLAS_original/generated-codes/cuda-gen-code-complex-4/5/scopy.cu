#include "chatblas_cuda.h"

__global__ void scopy_kernel( int n, float *x, float *y ) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = x[idx];
    }
}

void chatblas_scopy(int n, float *x, float *y) {
    float *d_x, *d_y;

    int size = n * sizeof(float);
    cudaMalloc((void**)&d_x, size);
    cudaMalloc((void**)&d_y, size);

    cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice);

    int blocks = (n + 255) / 256;
    scopy_kernel <<< blocks, 256 >>> (n, d_x, d_y);

    cudaMemcpy(y, d_y, size, cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_y);
}
