#include "chatblas_cuda.h"

__global__ void snrm2_kernel(int n, float *x, float *res) {
    __shared__ float partial_sum[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    float sum = 0.0;

    while (idx < n) {
        sum += x[idx] * x[idx];
        idx += blockDim.x * gridDim.x;
    }

    partial_sum[tid] = sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            partial_sum[tid] += partial_sum[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(res, partial_sum[0]);
    }
}

float chatblas_snrm2(int n, float *x) {
    float *d_x, *d_res;
    float result = 0.0;
    int size = n * sizeof(float);

    cudaMalloc((void **)&d_x, size);
    cudaMalloc((void **)&d_res, sizeof(float));
    cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice);
    cudaMemset(d_res, 0, sizeof(float));

    int numBlocks = (n + 256 - 1) / 256;
    snrm2_kernel<<<numBlocks, 256>>>(n, d_x, d_res);

    cudaMemcpy(&result, d_res, sizeof(float), cudaMemcpyDeviceToHost);
    result = sqrt(result);

    cudaFree(d_x);
    cudaFree(d_res);

    return result;
}