#include "chatblas_hip.h"

__global__ void snrm2_kernel(int n, float *x, float *res) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    __shared__ float sdata[256];

    float x_val = 0;
    if(idx < n) {
        x_val = x[idx];
        sdata[threadIdx.x] = x_val * x_val;
    }
    __syncthreads();

    for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    if(threadIdx.x == 0)
        atomicAdd(res, sqrtf(sdata[0]));
}

float chatblas_snrm2(int n, float *x) {
    float *d_x, *d_res;
    float *res = (float*)malloc(sizeof(float));
    *res = 0;

    hipMalloc((void**)&d_x, n*sizeof(float));
    hipMalloc((void**)&d_res, sizeof(float));

    hipMemcpy(d_x, x, n*sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_res, res, sizeof(float), hipMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid =(n + threadsPerBlock - 1) / threadsPerBlock;

    hipLaunchKernelGGL(snrm2_kernel, dim3(blocksPerGrid), dim3(threadsPerBlock), 0, 0, n, d_x, d_res);

    hipMemcpy(res, d_res, sizeof(float), hipMemcpyDeviceToHost);

    float result = *res;

    free(res);
    hipFree(d_x);
    hipFree(d_res);

    return result;
}
