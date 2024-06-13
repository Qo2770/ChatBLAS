#include "chatblas_hip.h"

__global__ void snrm2_kernel(int n, float *x, float *res) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    __shared__ float sum[256];

    sum[threadIdx.x] = 0;
    if(idx < n)
        sum[threadIdx.x] = x[idx] * x[idx];
    
    __syncthreads();

    for(int s = blockDim.x/2; s > 0; s >>= 1) {
        if(threadIdx.x < s)
            sum[threadIdx.x] += sum[threadIdx.x + s];
        
        __syncthreads();
    }

    if(threadIdx.x == 0)
        atomicAdd(res, sqrtf(sum[0]));
}

float chatblas_snrm2(int n, float *x) {
    float *d_x, *d_res;
    int blocks = (n + 255) / 256;
    float res;

    hipMalloc(&d_x, n * sizeof(float));
    hipMalloc(&d_res, sizeof(float));
    
    hipMemcpy(d_x, x, n * sizeof(float), hipMemcpyHostToDevice);
    hipMemset(d_res, 0, sizeof(float));

    snrm2_kernel<<<blocks, 256>>>(n, d_x, d_res);

    hipMemcpy(&res, d_res, sizeof(float), hipMemcpyDeviceToHost);
    
    hipFree(d_x);
    hipFree(d_res);

    return res;
}