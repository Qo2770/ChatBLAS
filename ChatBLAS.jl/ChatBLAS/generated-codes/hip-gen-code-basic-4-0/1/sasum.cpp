#include "chatblas_hip.h"

__global__ void sasum_kernel(int n, float *x, float *sum) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i<n) atomicAdd(sum, fabsf(x[i]));
}

float chatblas_sasum(int n, float *x) {
    float *d_x, *d_sum;
    float h_sum = 0.0;
    
    hipMalloc(&d_x, n*sizeof(float));
    hipMemcpy(d_x, x, n*sizeof(float), hipMemcpyHostToDevice);
    
    hipMalloc(&d_sum, sizeof(float));
    hipMemcpy(d_sum, &h_sum, sizeof(float), hipMemcpyHostToDevice);
    
    sasum_kernel<<<(n+255)/256, 256>>>(n, d_x, d_sum);
    
    hipMemcpy(&h_sum, d_sum, sizeof(float), hipMemcpyDeviceToHost);
    
    hipFree(d_x);
    hipFree(d_sum);
    
    return h_sum;
}
