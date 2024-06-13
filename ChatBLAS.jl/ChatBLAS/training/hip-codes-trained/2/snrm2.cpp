#include "chatblas_hip.h" 

__global__ void snrm2_kernel(int n, float *x, float *res) { int index = threadIdx.x + blockIdx.x * blockDim.x; extern __shared__ float sum[]; sum[index] = 0.0; if (index < n) { sum[index] = x[index] * x[index]; } for(int i = blockDim.x/2; i>0; i>>=1) { __syncthreads(); if(index < i) { sum[index] += sum[index + i]; } } if (index == 0) { *res = sqrt(sum[0]); } } 

float chatblas_snrm2(int n, float *x) { float *res, *d_x, *d_res; hipMalloc((void**)&d_x, n*sizeof(float)); hipMalloc((void**)&d_res, sizeof(float)); int blockSize = 256; int numBlocks = (n + blockSize - 1) / blockSize; hipMemcpy(d_x, x, n*sizeof(float), hipMemcpyHostToDevice); snrm2_kernel<<<numBlocks, blockSize>>>(n, d_x, d_res); res = (float*)malloc(sizeof(float)); hipMemcpy(res, d_res, sizeof(float), hipMemcpyDeviceToHost); float result = *res; free(res); hipFree(d_x); hipFree(d_res); return result; }
