#include "chatblas_cuda.h"

__global__ void sscal_kernel(int n, float a, float *x) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < n) {
        x[i] *= a;
    }
}

void chatblas_sscal(int n, float a, float *x) {
    const int blockSize = 256;
    const int numBlocks = (n + blockSize - 1) / blockSize;
    
    float *d_x;
    cudaMalloc((void **)&d_x, n * sizeof(float));
    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
    
    sscal_kernel<<<numBlocks, blockSize>>>(n, a, d_x);
    
    cudaMemcpy(x, d_x, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_x);
}
