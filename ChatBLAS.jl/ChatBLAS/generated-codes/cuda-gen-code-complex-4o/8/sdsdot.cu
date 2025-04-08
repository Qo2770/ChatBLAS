#include "chatblas_cuda.h"

__global__ void sdsdot_kernel( int n, float b, float *x, float *y, float *res ) {
    extern __shared__ double shared_mem[];
    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + tid;
    double sum = 0.0;
    
    for (int i = index; i < n; i += blockDim.x * gridDim.x) {
        sum += (double)x[i] * (double)y[i];
    }
    
    shared_mem[tid] = sum;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_mem[tid] += shared_mem[tid + stride];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(res, shared_mem[0]);
    }
}

float chatblas_sdsdot( int n, float b, float *x, float *y) {
    float *d_x, *d_y, *d_res;
    float h_res = 0.0f;
    cudaMalloc((void**)&d_x, n * sizeof(float));
    cudaMalloc((void**)&d_y, n * sizeof(float));
    cudaMalloc((void**)&d_res, sizeof(float));
    
    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_res, &h_res, sizeof(float), cudaMemcpyHostToDevice);
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    sdsdot_kernel<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(double)>>>(n, b, d_x, d_y, d_res);
    
    cudaMemcpy(&h_res, d_res, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_res);
    
    return h_res + b;
}