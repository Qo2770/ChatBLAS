#include "chatblas_cuda.h"

__global__ void sdsdot_kernel(int n, float b, float *x, float *y, float *res) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    float temp = 0.0f;
    if (index < n) {
        temp = __double2float_rn((double)x[index] * (double)y[index]);
    }
    atomicAdd(res, temp);
}

float chatblas_sdsdot(int n, float b, float *x, float *y) {
    int size = n * sizeof(float);
    float *d_x, *d_y, *d_res, *h_res;

    cudaMalloc((void **)&d_x, size);
    cudaMalloc((void **)&d_y, size);
    cudaMalloc((void **)&d_res, sizeof(float));

    h_res = (float *)malloc(sizeof(float));

    cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_res, &b, sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 1024;
    int numBlocks = (n + blockSize - 1) / blockSize;

    sdsdot_kernel<<<numBlocks, blockSize>>>(n, b, d_x, d_y, d_res);
    
    cudaMemcpy(h_res, d_res, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_res);

    float result = *h_res;
    free(h_res);

    return result;
}