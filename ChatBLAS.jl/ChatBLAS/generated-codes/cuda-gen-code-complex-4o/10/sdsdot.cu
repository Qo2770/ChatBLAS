#include "chatblas_cuda.h"

__global__ void sdsdot_kernel( int n, float b, float *x, float *y, float *res ) {
    extern __shared__ double sdata[];
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int s = blockDim.x / 2; 
    double sum = 0.0;
    
    while (tid < n) {
        sum += (double)x[tid] * (double)y[tid];
        tid += blockDim.x * gridDim.x;
    }

    sdata[threadIdx.x] = sum;
    __syncthreads();

    for (int stride = s; stride > 0; stride /= 2) { 
        if (threadIdx.x < stride) 
            sdata[threadIdx.x] += sdata[threadIdx.x + stride];
        __syncthreads();
    }
    
    if (threadIdx.x == 0)
        atomicAdd(res, sdata[0]);
}

float chatblas_sdsdot( int n, float b, float *x, float *y) {
    float *d_x, *d_y, *d_res;
    float result = 0.0;
    
    cudaMalloc((void**)&d_x, n * sizeof(float));
    cudaMalloc((void**)&d_y, n * sizeof(float));
    cudaMalloc((void**)&d_res, sizeof(float));
    
    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_res, &result, sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    
    sdsdot_kernel<<<gridSize, blockSize, blockSize * sizeof(double)>>>(n, b, d_x, d_y, d_res);
    
    cudaMemcpy(&result, d_res, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_res);
    
    return result + b;
}