#include "chatblas_hip.h"

__global__ void snrm2_kernel(int n, float *x, float *res) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    extern __shared__ float sdata[];

    float sum = 0.0f;

    while (tid < n) {
        sum += x[tid] * x[tid];
        tid += blockDim.x * gridDim.x;
    }

    sdata[threadIdx.x] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(res, sqrt(sdata[0]));
    }
}

float chatblas_snrm2(int n, float *x) {
    float *d_x, *d_res, h_res;
    cudaMalloc(&d_x, n * sizeof(float));
    cudaMalloc(&d_res, sizeof(float));
    
    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
    
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    
    snrm2_kernel<<<numBlocks, blockSize, blockSize * sizeof(float)>>>(n, d_x, d_res);
    
    cudaMemcpy(&h_res, d_res, sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(d_x);
    cudaFree(d_res);
    
    return h_res;
}