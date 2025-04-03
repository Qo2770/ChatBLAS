#include "chatblas_cuda.h"

__global__ void sdot_kernel(int n, float *x, float *y, float *res) {
    int id = blockIdx.x*blockDim.x + threadIdx.x;
    __shared__ float temp[256];
    temp[threadIdx.x] = (id < n) ? x[id] * y[id] : 0;
    __syncthreads();

    if (0 == threadIdx.x) {
        float sum_val = 0;
        for(int i = 0; i < 256; i++) {
            sum_val += temp[i];
        }
        atomicAdd(res, sum_val);
    }
}

float chatblas_sdot( int n, float *x, float *y) {
    float *x_device, *y_device, *res_device, res_host = 0;
    int size = n*sizeof(float);
    int block_size = 256;
    int grid_size = (n + block_size - 1)/ block_size;

    cudaMalloc((void**)&x_device, size);
    cudaMalloc((void**)&y_device, size);
    cudaMalloc((void**)&res_device, sizeof(float));

    cudaMemcpy(x_device, x, size, cudaMemcpyHostToDevice);
    cudaMemcpy(y_device, y, size, cudaMemcpyHostToDevice);
    cudaMemcpy(res_device, &res_host, sizeof(float), cudaMemcpyHostToDevice);

    sdot_kernel<<<grid_size,block_size>>>(n, x_device, y_device, res_device);

    cudaMemcpy(&res_host, res_device, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(x_device);
    cudaFree(y_device);
    cudaFree(res_device);

    return res_host;
}