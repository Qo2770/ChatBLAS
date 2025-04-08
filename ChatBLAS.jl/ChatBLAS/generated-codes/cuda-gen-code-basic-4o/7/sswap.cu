#include "chatblas_cuda.h"

__global__ void swap_kernel(int n, float *x, float *y) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        float temp = x[idx];
        x[idx] = y[idx];
        y[idx] = temp;
    }
}

void chatblas_sswap(int n, float *x, float *y) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    swap_kernel<<<blocksPerGrid, threadsPerBlock>>>(n, x, y);

    cudaDeviceSynchronize();
}
