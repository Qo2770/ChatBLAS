#include "chatblas_cuda.h"

__global__ void sdot_kernel(int n, float *x, float *y, float *res) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int index = tid;
    float sum = 0.0f;
    
    while (index < n) {
        sum += x[index] * y[index];
        index += blockDim.x * gridDim.x;
    }
    
    sdata[threadIdx.x] = sum;
    __syncthreads();
    
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0) {
        atomicAdd(res, sdata[0]);
    }
}

float chatblas_sdot(int n, float *x, float *y) {
    float *d_x, *d_y, *d_res;
    float h_res = 0.0f;
    cudaMalloc((void **)&d_x, n * sizeof(float));
    cudaMalloc((void **)&d_y, n * sizeof(float));
    cudaMalloc((void **)&d_res, sizeof(float));
    
    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_res, &h_res, sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    sdot_kernel<<<gridSize, blockSize, blockSize * sizeof(float)>>>(n, d_x, d_y, d_res);
    
    cudaMemcpy(&h_res, d_res, sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_res);
    
    return h_res;
}