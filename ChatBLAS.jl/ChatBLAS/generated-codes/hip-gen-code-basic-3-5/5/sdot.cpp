#include "hip/hip_runtime.h"

__global__ void sdot_kernel(int n, float *x, float *y, float *res) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    __shared__ float shared_res[256];
    
    if(tid < n) {
        shared_res[threadIdx.x] = x[tid] * y[tid];
    }
    __syncthreads();
    
    for(int stride = 128; stride > 0; stride /= 2) {
        if(threadIdx.x < stride) {
            shared_res[threadIdx.x] += shared_res[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    if(threadIdx.x == 0) {
        atomicAdd(res, shared_res[0]);
    }
}

float chatblas_sdot( int n, float *x, float *y) {
    float *d_x, *d_y, *d_res, result;
    
    hipMalloc(&d_x, n * sizeof(float));
    hipMalloc(&d_y, n * sizeof(float));
    hipMalloc(&d_res, sizeof(float));
    
    hipMemcpy(d_x, x, n * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_y, y, n * sizeof(float), hipMemcpyHostToDevice);
    
    sdot_kernel<<<1, 256>>>(n, d_x, d_y, d_res);
    
    hipMemcpy(&result, d_res, sizeof(float), hipMemcpyDeviceToHost);
    
    hipFree(d_x);
    hipFree(d_y);
    hipFree(d_res);
    
    return result;
}