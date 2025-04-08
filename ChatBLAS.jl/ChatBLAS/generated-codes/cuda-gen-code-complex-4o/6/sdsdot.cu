#include "chatblas_cuda.h"

__global__ void sdsdot_kernel( int n, float b, float *x, float *y, float *res ) {
    extern __shared__ double sdata[];
    int tid = threadIdx.x;
    int idx = blockDim.x * blockIdx.x + tid;
    double sum = 0.0;

    if (idx < n) {
        double a = static_cast<double>(x[idx]);
        double b = static_cast<double>(y[idx]);
        sum = a * b;
    }

    sdata[tid] = sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(res, sdata[0]);
    }
}

float chatblas_sdsdot( int n, float b, float *x, float *y) {
    float *d_x, *d_y, *d_res;
    float h_res = 0.0f;
    double final_res;

    cudaMalloc((void**)&d_x, n * sizeof(float));
    cudaMalloc((void**)&d_y, n * sizeof(float));
    cudaMalloc((void**)&d_res, sizeof(float));

    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_res, &h_res, sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    sdsdot_kernel<<<numBlocks, blockSize, blockSize * sizeof(double)>>>(n, b, d_x, d_y, d_res);

    cudaMemcpy(&h_res, d_res, sizeof(float), cudaMemcpyDeviceToHost);

    final_res = static_cast<double>(h_res) + static_cast<double>(b);

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_res);

    return static_cast<float>(final_res);
}