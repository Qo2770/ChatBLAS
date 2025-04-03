#include "chatblas_hip.h" 

__global__ void snrm2_kernel(int n, float *x, float *res) { int index = blockIdx.x * blockDim.x + threadIdx.x; extern __shared__ float sum[]; sum[index] = 0.0; if(index < n) { sum[index] = x[index] * x[index]; } __syncthreads(); for(int i = blockDim.x / 2; i > 0; i >>= 1) { if(index < i) { sum[index] += sum[index + i]; } __syncthreads(); } if(index == 0) { res[0] = sqrtf(sum[0]); } } 

float chatblas_snrm2(int n, float *x) { float *res, *d_x, *d_res; hipMalloc((void**)&d_x, n*sizeof(float)); hipMemcpy(d_x, x, n*sizeof(float), hipMemcpyHostToDevice); int blockSize = 256; int gridSize = (n + blockSize - 1) / blockSize; hipMalloc((void**)&d_res, 1 * sizeof(float)); snrm2_kernel<<<gridSize, blockSize>>>(n, d_x, d_res); res = (float*)malloc(1 * sizeof(float)); hipMemcpy(res, d_res, 1 * sizeof(float), hipMemcpyDeviceToHost); float norm = res[0]; free(res); hipFree(d_x); hipFree(d_res); return norm; }
