#include "chatblas_hip.h"

__device__ void warpReduce(volatile float *sdata, int tid) {
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}

__global__ void sdsdot_kernel(int n, float b, float *x, float *y, float *res) {
    __shared__ float temp[256];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    float sum = 0.0f;
    while (i < n) {
        sum += static_cast<float>(x[i]) * static_cast<float>(y[i]);
        i += blockDim.x * gridDim.x;
    }
    
    temp[tid] = sum;
    __syncthreads();
    
    if (tid < 128) temp[tid] += temp[tid + 128];
    __syncthreads();
    if (tid < 64) temp[tid] += temp[tid + 64];
    __syncthreads();
    
    if (tid < 32) warpReduce(temp, tid);

    if (tid == 0) atomicAdd(res, static_cast<float>(b) + static_cast<float>(temp[0]));
}

float chatblas_sdsdot(int n, float b, float *x, float *y) {
    float *d_x, *d_y, *d_res;
    float h_res = 0.0f;

    hipMalloc(&d_x, n * sizeof(float));
    hipMalloc(&d_y, n * sizeof(float));
    hipMalloc(&d_res, sizeof(float));

    hipMemcpy(d_x, x, n * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_y, y, n * sizeof(float), hipMemcpyHostToDevice);

    sdsdot_kernel<<<n/256, 256>>>(n, b, d_x, d_y, d_res);

    hipMemcpy(&h_res, d_res, sizeof(float), hipMemcpyDeviceToHost);

    hipFree(d_x);
    hipFree(d_y);
    hipFree(d_res);

    return h_res;
}