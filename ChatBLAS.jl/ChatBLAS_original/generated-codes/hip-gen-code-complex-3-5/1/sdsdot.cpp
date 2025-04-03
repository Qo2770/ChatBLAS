#include "chatblas_hip.h"

__global__ void sdsdot_kernel( int n, float b, float *x, float *y, float *res ) {
    __shared__ float partialSum[256];
    
    int tid = threadIdx.x;
    int blockSize = blockDim.x;
    int blockId = blockIdx.x;
    int i = tid + blockId * blockSize;
    
    float sum = 0.0f;
    
    while(i < n) {
        sum += (float)x[i] * (float)y[i];
        i += gridDim.x * blockDim.x;
    }
    
    partialSum[tid] = sum;
    __syncthreads();
    
    // Reduction within block
    if(blockSize >= 256 && tid < 128) {
        partialSum[tid] += partialSum[tid + 128];
    }
    __syncthreads();
    
    if(blockSize >= 128 && tid < 64) {
        partialSum[tid] += partialSum[tid + 64];
    }
    __syncthreads();
    
    if(tid < 32) {
        // Unrolling warp
        partialSum[tid] += partialSum[tid + 32];
        partialSum[tid] += partialSum[tid + 16];
        partialSum[tid] += partialSum[tid + 8];
        partialSum[tid] += partialSum[tid + 4];
        partialSum[tid] += partialSum[tid + 2];
        partialSum[tid] += partialSum[tid + 1];
    }
    
    if(tid == 0) {
        atomicAdd(res, (double)partialSum[0] + (double)b);
    }
}

float chatblas_sdsdot( int n, float b, float *x, float *y) {
    float *d_x, *d_y, *d_res;
    float h_res;
    
    hipMalloc(&d_x, n * sizeof(float));
    hipMalloc(&d_y, n * sizeof(float));
    hipMalloc(&d_res, sizeof(float));
    
    hipMemcpy(d_x, x, n * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_y, y, n * sizeof(float), hipMemcpyHostToDevice);
    
    sdsdot_kernel<<<blocks, threads>>>(n, b, d_x, d_y, d_res);
    
    hipMemcpy(&h_res, d_res, sizeof(float), hipMemcpyDeviceToHost);
    
    hipFree(d_x);
    hipFree(d_y);
    hipFree(d_res);
    
    return h_res;
}