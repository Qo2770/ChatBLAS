#include "chatblas_cuda.h"

__global__ void chatblas_scopy_cuda(int n, float *x, float *y) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n) {
        y[index] = x[index];
    }
}

void chatblas_scopy(int n, float *x, float *y) {
    float* d_x;
    float* d_y;

    cudaMalloc(&d_x, n*sizeof(float)); 
    cudaMalloc(&d_y, n*sizeof(float));

    cudaMemcpy(d_x, x, n*sizeof(float), cudaMemcpyHostToDevice);

    chatblas_scopy_cuda<<<(n+255)/256, 256>>>(n, d_x, d_y);

    cudaMemcpy(y, d_y, n*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_y);
}
