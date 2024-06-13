#include "chatblas_hip.h"

__global__ void snrm2_kernel(int n, float *x, float *res) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    
    if (tid < n) {
        sum = x[tid] * x[tid];
    }
    
    sum = sqrtf(blockReduceSum(sum));
    
    if (threadIdx.x == 0) {
        atomicAdd(res, sum);
    }
}

float chatblas_snrm2(int n, float *x) {
    float *d_x, *d_res;
    float h_res = 0.0f;
    
    hipMalloc(&d_x, n * sizeof(float));
    hipMalloc(&d_res, sizeof(float));
    
    hipMemcpy(d_x, x, n * sizeof(float), hipMemcpyHostToDevice);
    
    snrm2_kernel<<<(n+255)/256, 256>>>(n, d_x, d_res);
    
    hipMemcpy(&h_res, d_res, sizeof(float), hipMemcpyDeviceToHost);
    
    hipFree(d_x);
    hipFree(d_res);
    
    return h_res;
}
