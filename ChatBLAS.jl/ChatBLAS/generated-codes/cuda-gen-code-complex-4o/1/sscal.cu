#include "chatblas_cuda.h"

__global__ void sscal_kernel( int n, float a , float *x ) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        x[idx] *= a;
    }
}

void chatblas_sscal( int n, float a, float *x) {
    float *d_x;
    size_t size = n * sizeof(float);
    
    cudaMalloc((void**)&d_x, size);
    cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice);
    
    int blockSize = 256; // example block size
    int numBlocks = (n + blockSize - 1) / blockSize;
    
    sscal_kernel<<<numBlocks, blockSize>>>(n, a, d_x);
    
    cudaMemcpy(x, d_x, size, cudaMemcpyDeviceToHost);
    cudaFree(d_x);
}