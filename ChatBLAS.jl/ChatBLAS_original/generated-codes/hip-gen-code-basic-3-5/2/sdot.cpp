#include "hip/hip_runtime.h"

__global__ void sdot_kernel(int n, float *x, float *y, float *res) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    __shared__ float temp[256];
    temp[threadIdx.x] = x[idx] * y[idx];
    
    for(int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        __syncthreads();
        if(threadIdx.x < stride) {
            temp[threadIdx.x] += temp[threadIdx.x + stride];
        }
    }
    
    if(threadIdx.x == 0) {
        atomicAdd(res, temp[0]);
    }
}

float chatblas_sdot(int n, float *x, float *y) {
    float *d_x, *d_y, *d_res;
    float h_res = 0;
    
    hipMalloc(&d_x, n * sizeof(float));
    hipMalloc(&d_y, n * sizeof(float));
    hipMalloc(&d_res, sizeof(float));
    
    hipMemcpy(d_x, x, n * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_y, y, n * sizeof(float), hipMemcpyHostToDevice);
    
    hipLaunchKernelGGL(sdot_kernel, dim3(1), dim3(n), 0, 0, n, d_x, d_y, d_res);
    
    hipMemcpy(&h_res, d_res, sizeof(float), hipMemcpyDeviceToHost);
    
    hipFree(d_x);
    hipFree(d_y);
    hipFree(d_res);
    
    return h_res;
}