#include "chatblas_cuda.h"

__global__ void scopy_kernel( int n, float *x, float *y ) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i < n ) y[i] = x[i];
}

void chatblas_scopy(int n, float *x, float *y) {
    float* x_d;
    float* y_d;

    cudaMalloc((void **)&x_d, n*sizeof(float));
    cudaMalloc((void **)&y_d, n*sizeof(float));
  
    cudaMemcpy(x_d, x, n*sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    scopy_kernel<<<numBlocks, blockSize>>>(n, x_d, y_d);
  
    cudaMemcpy(y, y_d, n*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(x_d);
    cudaFree(y_d);
}