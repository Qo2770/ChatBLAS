#include "chatblas_cuda.h"

__global__ void sdot_kernel(int n, float *x, float *y, float *res) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    __shared__ float temp[BLOCK_SIZE];

    temp[threadIdx.x] = 0;
    if(idx < n) {
        temp[threadIdx.x] = x[idx] * y[idx];
    }

    __syncthreads();

    if(threadIdx.x == 0) {
        float sum = 0;
        for(int i=0; i<blockDim.x; i++) {
            sum += temp[i];
        }
        atomicAdd(res, sum);
    }
}

float chatblas_sdot( int n, float *x, float *y) {
    int size = n * sizeof(float);
    float *d_x, *d_y, *d_res;
    float *res = (float*)malloc(sizeof(float));

    cudaMalloc(&d_x, size);
    cudaMalloc(&d_y, size);
    cudaMalloc(&d_res, sizeof(float));

    cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, size, cudaMemcpyHostToDevice);
    cudaMemset(d_res, 0, sizeof(float));

    dim3 dimBlock(BLOCK_SIZE);
    dim3 dimGrid((n + dimBlock.x - 1) / dimBlock.x);

    sdot_kernel<<<dimGrid, dimBlock>>>(n, d_x, d_y, d_res);
    cudaMemcpy(res, d_res, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_res);
    cudaFree(d_x);
    cudaFree(d_y);

    float dot_product = *res;
    free(res);

    return dot_product;
}