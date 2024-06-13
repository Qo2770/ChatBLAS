#include "chatblas_cuda.h"

__global__ void sscal_kernel( int n, float a , float *x ) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        x[i] = a * x[i];
    }
}

void chatblas_sscal( int n, float a, float *x) {
    float *x_d;
    int size = n * sizeof(float);
    cudaMalloc((void **)&x_d, size);
    cudaMemcpy(x_d, x, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    sscal_kernel<<<blocksPerGrid, threadsPerBlock>>>(n, a, x_d);

    cudaMemcpy(x, x_d, size, cudaMemcpyDeviceToHost);
    cudaFree(x_d);
}
