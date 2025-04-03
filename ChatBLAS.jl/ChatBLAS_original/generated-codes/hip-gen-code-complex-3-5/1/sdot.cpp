#include "chatblas_hip.h"

__global__ void sdot_kernel(int n, float *x, float *y, float *res) {
    __shared__ float shared_mem[256];
    float partial_sum = 0.0f;
    int tid = threadIdx.x;
    
    for (int i = tid; i < n; i += blockDim.x) {
        partial_sum += x[i] * y[i];
    }
    
    shared_mem[tid] = partial_sum;
    __syncthreads();
    
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_mem[tid] += shared_mem[tid + stride];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(res, shared_mem[0]);
    }
}

float chatblas_sdot(int n, float *x, float *y) {
    float *d_x, *d_y, *d_res;
    float h_res;
    
    hipMalloc(&d_x, n * sizeof(float));
    hipMalloc(&d_y, n * sizeof(float));
    hipMalloc(&d_res, sizeof(float));
    
    hipMemcpy(d_x, x, n * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_y, y, n * sizeof(float), hipMemcpyHostToDevice);
    
    sdot_kernel<<<1, 256>>>(n, d_x, d_y, d_res);
    
    hipMemcpy(&h_res, d_res, sizeof(float), hipMemcpyDeviceToHost);
    
    hipFree(d_x);
    hipFree(d_y);
    hipFree(d_res);
    
    return h_res;
}