#include "chatblas_cuda.h"

__global__ void snrm2_kernel(int n, float *x, float *res) {
    __shared__ float cache[256];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    float temp = 0.0;

    while (tid < n) {
        temp += x[tid] * x[tid];
        tid += blockDim.x * gridDim.x;
    }

    cache[threadIdx.x] = temp;

    __syncthreads();

    int i = blockDim.x / 2;
    while (i != 0) {
        if (threadIdx.x < i) {
            cache[threadIdx.x] += cache[threadIdx.x + i];
        }
        __syncthreads();
        i /= 2;
    }

    if (threadIdx.x == 0) {
        atomicAdd(res, cache[0]);
    }
}

float chatblas_snrm2(int n, float *x) {
    float *d_x, *d_res;
    float h_res = 0.0;
    cudaMalloc((void**)&d_x, n * sizeof(float));
    cudaMalloc((void**)&d_res, sizeof(float));
    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_res, &h_res, sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    snrm2_kernel<<<blocksPerGrid, threadsPerBlock>>>(n, d_x, d_res);

    cudaMemcpy(&h_res, d_res, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_x);
    cudaFree(d_res);

    return sqrtf(h_res);
}