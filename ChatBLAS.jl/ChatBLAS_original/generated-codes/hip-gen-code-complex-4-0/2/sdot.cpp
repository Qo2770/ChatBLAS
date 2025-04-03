#include "chatblas_hip.h"

__global__ void sdot_kernel(int n, float *x, float *y, float *res) {
    __shared__ float sdata[256];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = (i < n) ? x[i] * y[i] : 0;
    __syncthreads();
    for(unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if(tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if(tid == 0) {
        atomicAdd(res, sdata[0]);
    }
}

float chatblas_sdot( int n, float *x, float *y) {
    float *d_x, *d_y, *d_res, result;
    int blockSize = 256;
    int gridSize = (n-1)/blockSize + 1;

    hipMalloc(&d_x, n * sizeof(float));
    hipMalloc(&d_y, n * sizeof(float));
    hipMalloc(&d_res, sizeof(float));

    hipMemcpy(d_x, x, n*sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_y, y, n*sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_res, &result, sizeof(float), hipMemcpyHostToDevice);

    sdot_kernel<<<gridSize, blockSize>>>(n, d_x, d_y, d_res);
    hipDeviceSynchronize();

    hipMemcpy(&result, d_res, sizeof(float), hipMemcpyDeviceToHost);

    hipFree(d_x); 
    hipFree(d_y); 
    hipFree(d_res); 

    return result; 
}