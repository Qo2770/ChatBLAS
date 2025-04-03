#include "chatblas_cuda.h"

__global__ void sscal_kernel(int n, float a, float *x) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n) {
        x[tid] *= a;
    }
}

void chatblas_sscal(int n, float a, float *x) {
    float *d_x;
    
    cudaMalloc((void**)&d_x, n * sizeof(float));
    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
    
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    
    sscal_kernel<<<gridSize, blockSize>>>(n, a, d_x);
    
    cudaMemcpy(x, d_x, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_x);
}
