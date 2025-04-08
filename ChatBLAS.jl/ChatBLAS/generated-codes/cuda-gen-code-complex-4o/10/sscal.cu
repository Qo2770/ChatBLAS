#include "chatblas_cuda.h"

__global__ void sscal_kernel(int n, float a, float *x) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        x[idx] *= a;
    }
}

void chatblas_sscal(int n, float a, float *x) {
    float *d_x;
    size_t size = n * sizeof(float);

    cudaMalloc((void**)&d_x, size);
    cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    sscal_kernel<<<blocksPerGrid, threadsPerBlock>>>(n, a, d_x);

    cudaMemcpy(x, d_x, size, cudaMemcpyDeviceToHost);
    cudaFree(d_x);
}