#include "chatblas_cuda.h"

__global__ void snrm2_kernel(int n, float *x, float *res) {
    __shared__ float cache[256];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;
    float temp = 0;
    while (tid < n) {
        temp += x[tid] * x[tid];
        tid += blockDim.x * gridDim.x;
    }
    cache[cacheIndex] = temp;
    __syncthreads();
    int i = blockDim.x / 2;
    while (i != 0) {
        if (cacheIndex < i) {
            cache[cacheIndex] += cache[cacheIndex + i];
        }
        __syncthreads();
        i /= 2;
    }
    if (cacheIndex == 0) {
        atomicAdd(res, cache[0]);
    }
}

float chatblas_snrm2(int n, float *x) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    float *d_x, *d_res, norm = 0.0;
    cudaMalloc((void**)&d_x, n * sizeof(float));
    cudaMalloc((void**)&d_res, sizeof(float));
    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_res, &norm, sizeof(float), cudaMemcpyHostToDevice);
    snrm2_kernel<<<blocksPerGrid, threadsPerBlock>>>(n, d_x, d_res);
    cudaMemcpy(&norm, d_res, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_x);
    cudaFree(d_res);
    return sqrt(norm);
}