#include "chatblas_cuda.h" 

__global__ void snrm2_kernel(int n, float *x, float *res) { int index = blockIdx.x * blockDim.x + threadIdx.x; int stride = blockDim.x * gridDim.x; float local_res = 0.0f; for (int i = index; i < n; i += stride) { local_res += x[i] * x[i]; } atomicAdd(res, local_res); } 

float chatblas_snrm2(int n, float *x) { int blockSize = 256; int numBlocks = (n + blockSize - 1) / blockSize; float *d_x, *d_res; float res = 0.0f; cudaMalloc((void **)&d_x, n * sizeof(float)); cudaMalloc((void **)&d_res, sizeof(float)); cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice); cudaMemcpy(d_res, &res, sizeof(float), cudaMemcpyHostToDevice); snrm2_kernel<<<numBlocks, blockSize>>>(n, d_x, d_res); cudaMemcpy(&res, d_res, sizeof(float), cudaMemcpyDeviceToHost); res = sqrt(res); cudaFree(d_x); cudaFree(d_res); return res; }
