#include "chatblas_cuda.h"

__global__ void sdsdot_kernel(int n, float b, float *x, float *y, float *res) {
    extern __shared__ double shared_mem[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    double sum = 0.0;

    while (idx < n) {
        sum += (double)x[idx] * (double)y[idx];
        idx += blockDim.x * gridDim.x;
    }

    shared_mem[tid] = sum;
    __syncthreads();

    if (tid == 0) {
        double blockSum = 0.0;
        for (int i = 0; i < blockDim.x; i++) {
            blockSum += shared_mem[i];
        }
        atomicAdd(res, blockSum);
    }
}

float chatblas_sdsdot(int n, float b, float *x, float *y) {
    float *d_x, *d_y, *d_res;
    float h_res = 0.0f;
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    cudaMalloc((void**)&d_x, n * sizeof(float));
    cudaMalloc((void**)&d_y, n * sizeof(float));
    cudaMalloc((void**)&d_res, sizeof(float));
    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_res, &h_res, sizeof(float), cudaMemcpyHostToDevice);

    sdsdot_kernel<<<numBlocks, blockSize, blockSize * sizeof(double)>>>(n, b, d_x, d_y, d_res);
    cudaMemcpy(&h_res, d_res, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_res);

    return h_res + b;
}