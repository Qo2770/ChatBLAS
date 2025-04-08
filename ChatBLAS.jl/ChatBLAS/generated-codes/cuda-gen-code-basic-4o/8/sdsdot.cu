#include "chatblas_cuda.h"
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void dot_product_kernel(float *d_x, float *d_y, double *d_partial_sums, int n) {
    extern __shared__ double sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = (i < n) ? (double)(d_x[i]) * (double)(d_y[i]) : 0.0;

    __syncthreads();

    // Parallel reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        d_partial_sums[blockIdx.x] = sdata[0];
    }
}

float chatblas_sdsdot(int n, float b, float *x, float *y) {
    float *d_x, *d_y;
    double *d_partial_sums;
    double partial_sums_host[256]; // Assume max 256 blocks for now
    double result = 0.0;

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    cudaMalloc((void **)&d_x, n * sizeof(float));
    cudaMalloc((void **)&d_y, n * sizeof(float));
    cudaMalloc((void **)&d_partial_sums, blocksPerGrid * sizeof(double));

    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, n * sizeof(float), cudaMemcpyHostToDevice);

    dot_product_kernel<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(double)>>>(d_x, d_y, d_partial_sums, n);

    cudaMemcpy(partial_sums_host, d_partial_sums, blocksPerGrid * sizeof(double), cudaMemcpyDeviceToHost);

    for (int i = 0; i < blocksPerGrid; i++) {
        result += partial_sums_host[i];
    }

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_partial_sums);

    return (float)(result + (double)b);
}
