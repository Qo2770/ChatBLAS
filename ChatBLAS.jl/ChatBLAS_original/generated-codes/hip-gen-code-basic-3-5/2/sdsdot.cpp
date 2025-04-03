#include "chatblas_hip.h"

__global__ void sdsdot_kernel( int n, float b, float *x, float *y, float *res ) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    __shared__ double sdata[256];
    double temp = 0.0;
    
    while(tid < n) {
        temp += (double)x[tid] * (double)y[tid];
        tid += blockDim.x * gridDim.x;
    }
    
    sdata[threadIdx.x] = temp;
    __syncthreads();
    
    for (int s = 1; s < blockDim.x; s *= 2) {
        if (threadIdx.x % (2 * s) == 0) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0) {
        atomicAdd(res, sdata[0] + (double)b);
    }
}

float chatblas_sdsdot( int n, float b, float *x, float *y) {
    float *d_x, *d_y, *d_res;
    float h_res = 0.0;
    
    hipMalloc(&d_x, n * sizeof(float));
    hipMalloc(&d_y, n * sizeof(float));
    hipMalloc(&d_res, sizeof(float));
    
    hipMemcpy(d_x, x, n * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_y, y, n * sizeof(float), hipMemcpyHostToDevice);
    
    hipLaunchKernelGGL(sdsdot_kernel, dim3(256), dim3(256), 0, 0, n, b, d_x, d_y, d_res);
    
    hipMemcpy(&h_res, d_res, sizeof(float), hipMemcpyDeviceToHost);
    
    hipFree(d_x);
    hipFree(d_y);
    hipFree(d_res);
    
    return h_res;
}