#include "chatblas_hip.h"

__global__ void snrm2_kernel( int n, float *x, float *res){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    
    for(int j=i;j<n;j+=blockDim.x*gridDim.x){
        sum += x[j] * x[j];
    }
    
    atomicAdd(res, sum);
}

float chatblas_snrm2(int n, float *x){
    float *d_x, *d_res;
    float res = 0.0f;
    
    hipMalloc(&d_x, n * sizeof(float));
    hipMalloc(&d_res, sizeof(float));
    
    hipMemcpy(d_x, x, n * sizeof(float), hipMemcpyHostToDevice);
    
    hipLaunchKernelGGL(snrm2_kernel, dim3(256), dim3(256), 0, 0, n, d_x, d_res);
    
    hipMemcpy(&res, d_res, sizeof(float), hipMemcpyDeviceToHost);
    
    hipFree(d_x);
    hipFree(d_res);
    
    return sqrt(res);
}