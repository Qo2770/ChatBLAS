#include "chatblas_hip.h"

__global__ void sdot_kernel(int n, float *x, float *y, float *res) {
    __shared__ float sdata[CHATBLAS_HIP_THREADS_PER_BLOCK];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = (i < n) ? x[i] * y[i] : 0;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0)
        atomicAdd(res, sdata[0]);
}

float chatblas_sdot(int n, float *x, float *y) {
    float *d_x, *d_y, *d_res, h_res;
    int blocks, threads = CHATBLAS_HIP_THREADS_PER_BLOCK;
    blocks = (n + threads - 1) / threads;

    hipMalloc((void **)&d_x, n*sizeof(float));
    hipMalloc((void **)&d_y, n*sizeof(float));
    hipMalloc((void **)&d_res, sizeof(float));
    
    hipMemcpy(d_x, x, n*sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_y, y, n*sizeof(float), hipMemcpyHostToDevice);
    hipMemset(d_res, 0, sizeof(float));
    
    sdot_kernel<<<blocks, threads>>>(n, d_x, d_y, d_res);
    
    hipMemcpy(&h_res, d_res, sizeof(float), hipMemcpyDeviceToHost);
  
    hipFree(d_x);
    hipFree(d_y);
    hipFree(d_res);
    
    return h_res;
}