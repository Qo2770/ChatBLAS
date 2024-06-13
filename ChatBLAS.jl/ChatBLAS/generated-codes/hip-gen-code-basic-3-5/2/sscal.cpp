#include "chatblas_hip.h"

__global__ void sscal_kernel( int n, float a, float *x ) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        x[tid] = a * x[tid];
    }
}

void chatblas_sscal( int n, float a, float *x) {
    float *d_x;
    hipMalloc(&d_x, n * sizeof(float));
    hipMemcpy(d_x, x, n * sizeof(float), hipMemcpyHostToDevice);

    dim3 threadsPerBlock(256);
    dim3 numBlocks((n + 255) / 256);

    sscal_kernel<<<numBlocks, threadsPerBlock>>>(n, a, d_x);

    hipMemcpy(x, d_x, n * sizeof(float), hipMemcpyDeviceToHost);
    hipFree(d_x);
}