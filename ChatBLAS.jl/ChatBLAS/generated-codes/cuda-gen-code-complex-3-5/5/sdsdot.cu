#include "chatblas_cuda.h"

__global__ void sdsdot_kernel(int n, float b, float *x, float *y, float *res) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    double sum = 0.0;
    for (int i = tid; i < n; i += stride) {
        sum += (double)x[i] * (double)y[i];
    }

    atomicAdd(res, (float)sum);
}

float chatblas_sdsdot(int n, float b, float *x, float *y) {
    float *d_x, *d_y, *d_res;
    float h_res = 0.0;

    cudaMalloc((void**)&d_x, sizeof(float) * n);
    cudaMalloc((void**)&d_y, sizeof(float) * n);
    cudaMalloc((void**)&d_res, sizeof(float));

    cudaMemcpy(d_x, x, sizeof(float) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, sizeof(float) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_res, &h_res, sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    sdsdot_kernel<<<numBlocks, blockSize>>>(n, b, d_x, d_y, d_res);

    cudaMemcpy(&h_res, d_res, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_res);

    return h_res + b;
}
