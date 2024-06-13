#include "chatblas_cuda.h"

__global__ void snrm2_kernel(int n, float *x, float *res) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        atomicAdd(res, x[idx] * x[idx]);
    }
}

float chatblas_snrm2(int n, float *x) {
    const int blocks = (n + 255) / 256;
    const int threads = min(n, 256);
    float *d_x, *d_res, h_res = 0;
    cudaMalloc((void **)&d_x, n * sizeof(float));
    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void **)&d_res, sizeof(float));
    cudaMemcpy(d_res, &h_res, sizeof(float), cudaMemcpyHostToDevice);
    snrm2_kernel<<<blocks, threads>>>(n, d_x, d_res);
    cudaMemcpy(&h_res, d_res, sizeof(float), cudaMemcpyDeviceToHost);
    h_res = sqrt(h_res);
    cudaFree(d_x);
    cudaFree(d_res);
    return h_res;
}