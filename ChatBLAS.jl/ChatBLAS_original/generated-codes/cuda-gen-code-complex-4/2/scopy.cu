#include "chatblas_cuda.h"

__global__ void scopy_kernel( int n, float *x, float *y ) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n) {
        y[idx] = x[idx];
    }
}

void chatblas_scopy(int n, float *x, float *y) {
    float *x_gpu, *y_gpu;

    cudaMalloc((void**) &x_gpu, n*sizeof(float));
    cudaMalloc((void**) &y_gpu, n*sizeof(float));

    cudaMemcpy(x_gpu, x, n*sizeof(float), cudaMemcpyHostToDevice);

    int blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    scopy_kernel<<<blocks, THREADS_PER_BLOCK>>>(n, x_gpu, y_gpu);

    cudaMemcpy(y, y_gpu, n*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(x_gpu);
    cudaFree(y_gpu);
}