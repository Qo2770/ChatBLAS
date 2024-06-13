#include "chatblas_hip.h"

__global__ void sasum_kernel(int n, float *x, float *sum) {
    __shared__ float partSum[256];
    int t = threadIdx.x;
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    partSum[t] = (i < n)? x[i] : 0.0;
    for(int stride=1; stride<blockDim.x; stride*=2) {
       __syncthreads();
       if(t%(2*stride)==0) {
          partSum[t] += partSum[t + stride];
       }
    }
    if(t == 0) {
       sum[blockIdx.x] += partSum[0];
    }
}


float chatblas_sasum(int n, float *x) {
    float *x_d;
    float *sum_d, sum_h = 0.0;
    size_t size = n*sizeof(float);

    HIP_CALL(hipMalloc(&x_d, size));
    HIP_CALL(hipMemcpy(x_d, x, size, hipMemcpyHostToDevice));

    HIP_CALL(hipMalloc(&sum_d, size));
    HIP_CALL(hipMemset(sum_d, 0, size));

    dim3 blockSize(256);
    dim3 grid((n/blockSize.x)+1);

    hipLaunchKernelGGL(sasum_kernel, dim3(grid), dim3(blockSize), 0, 0, n, x_d, sum_d);
    
    hipMemcpy(&sum_h, sum_d, size, hipMemcpyDeviceToHost);

    hipFree(x_d);
    hipFree(sum_d);

    return sum_h;
}