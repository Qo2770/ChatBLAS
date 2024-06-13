#include "chatblas_hip.h"

__global__ void sasum_kernel(int n, float *x, float *sum) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
  
    if(tid < n){
        atomicAdd(sum, fabs(x[tid]));
    }
}

float chatblas_sasum(int n, float *x) {
    float *d_x, *d_sum;
    float result = 0.0f;
  
    hipMalloc(&d_x, n * sizeof(float));
    hipMalloc(&d_sum, sizeof(float));
  
    hipMemcpy(d_x, x, n * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_sum, &result, sizeof(float), hipMemcpyHostToDevice);
  
    sasum_kernel<<< (n + 255) / 256, 256>>>(n, d_x, d_sum);
  
    hipMemcpy(&result, d_sum, sizeof(float), hipMemcpyDeviceToHost);
    
    hipFree(d_x);
    hipFree(d_sum);
  
    return result;
}