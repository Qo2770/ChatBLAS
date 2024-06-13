#include "chatblas_cuda.h"

__global__ void sscal_kernel( int n, float a , float *x ) {
    int indx = threadIdx.x + blockIdx.x * blockDim.x;
    if (indx < n) 
        x[indx] *= a;
}

void chatblas_sscal( int n, float a, float *x) {

    float *d_x;
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    cudaMalloc((void**)&d_x, n * sizeof(float));
    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);

    sscal_kernel<<<numBlocks, blockSize>>>(n, a, d_x);

    cudaMemcpy(x, d_x, n *sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_x);
}