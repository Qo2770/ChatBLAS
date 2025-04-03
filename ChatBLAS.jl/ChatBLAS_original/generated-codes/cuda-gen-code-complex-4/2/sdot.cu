#include "chatblas_cuda.h"

__global__ void sdot_kernel(int n, float *x, float *y, float *res) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    __shared__ float temp_res[256];
    temp_res[threadIdx.x] = (index < n)? x[index] * y[index] : 0;

    __syncthreads();

    if (threadIdx.x == 0) {
        float sum = 0;
        for (int i = 0; i < 256; i++) {
            sum += temp_res[i];
        }
        atomicAdd(res, sum);
    }
}

float chatblas_sdot(int n, float *x, float *y) {
    float *dev_x = 0;
    float *dev_y = 0;
    float *dev_res = 0;
    float res = 0;

    cudaMalloc((void**)&dev_x, n * sizeof(float));
    cudaMalloc((void**)&dev_y, n * sizeof(float));
    cudaMalloc((void**)&dev_res, sizeof(float));

    cudaMemcpy(dev_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_y, y, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_res, &res, sizeof(float), cudaMemcpyHostToDevice);

    int blocks = (n + 255) / 256;
    sdot_kernel <<< blocks, 256 >>> (n, dev_x, dev_y, dev_res);

    cudaMemcpy(&res, dev_res, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dev_x);
    cudaFree(dev_y);
    cudaFree(dev_res);

    return res;
}