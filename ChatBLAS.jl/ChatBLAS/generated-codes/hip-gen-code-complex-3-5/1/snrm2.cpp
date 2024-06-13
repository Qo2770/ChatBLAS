#include "chatblas_hip.h"
__global__ void snrm2_kernel(int n, float *x, float *res) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ float sdata[256];
    
    float temp = 0.0;
    
    if(idx < n) {
        temp = x[idx] * x[idx];
    }
    
    sdata[threadIdx.x] = temp;
    __syncthreads();
    
    for(int s = 1; s < blockDim.x; s *= 2) {
        if(threadIdx.x % (2 * s) == 0) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    if(threadIdx.x == 0) {
        atomicAdd(res, sdata[0]);
    }
}

float chatblas_snrm2(int n, float *x) {
    float *d_x, *d_res;
    float res;
    float host_res = 0.0;
    
    hipMalloc(&d_x, n * sizeof(float));
    hipMalloc(&d_res, sizeof(float));
    
    hipMemcpy(d_x, x, n * sizeof(float), hipMemcpyHostToDevice);
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    snrm2_kernel<<<blocksPerGrid, threadsPerBlock>>>(n, d_x, d_res);
    
    hipMemcpy(&res, d_res, sizeof(float), hipMemcpyDeviceToHost);
    
    host_res = sqrt(res);
    
    hipFree(d_x);
    hipFree(d_res);
    
    return host_res;
}