#include "hip/hip_runtime.h"

__global__ void snrm2_kernel( int n, float *x, float *res) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    float sum = 0.0f;
    
    if(index < n){
        sum = x[index] * x[index];
    }

    sum = hipBlockReduceSum(sum);

    if(threadIdx.x == 0){
        atomicAdd(res, sum);
    }
}

float chatblas_snrm2(int n, float *x) {
    float *d_x, *d_res, h_res;
    size_t size = n * sizeof(float);

    hipMalloc(&d_x, size);
    hipMalloc(&d_res, sizeof(float));

    hipMemcpy(d_x, x, size, hipMemcpyHostToDevice);
    
    hipLaunchKernelGGL(snrm2_kernel, dim3((n + 255)/256), dim3(256), 0, 0, n, d_x, d_res);

    hipMemcpy(&h_res, d_res, sizeof(float), hipMemcpyDeviceToHost);

    hipFree(d_x);
    hipFree(d_res);

    return sqrt(h_res);
}