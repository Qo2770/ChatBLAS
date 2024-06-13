#include "chatblas_cuda.h"

__global__ void snrm2_kernel(int n, float *x, float *res) {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    float sum = 0.0f;
    for (int i = thread_id; i < n; i += stride) {
        sum += x[i] * x[i];
    }

    __shared__ float shared_sum[256];
    shared_sum[threadIdx.x] = sum;
    __syncthreads();

    int i = blockDim.x / 2;
    while (i != 0) {
        if (threadIdx.x < i) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + i];
        }
        __syncthreads();
        i /= 2;
    }

    if (threadIdx.x == 0) {
        atomicAdd(res, shared_sum[0]);
    }
}

float chatblas_snrm2(int n, float *x) {
    float *gpu_x, *gpu_res;

    cudaMalloc((void **)&gpu_x, n * sizeof(float));
    cudaMemcpy(gpu_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void **)&gpu_res, sizeof(float));
    cudaMemset(gpu_res, 0, sizeof(float));

    int threads_per_block = 256;
    int blocks_per_grid = (n + threads_per_block - 1) / threads_per_block;

    snrm2_kernel<<<blocks_per_grid, threads_per_block>>>(n, gpu_x, gpu_res);
    cudaDeviceSynchronize();

    float res;
    cudaMemcpy(&res, gpu_res, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(gpu_x);
    cudaFree(gpu_res);

    return sqrt(res);
}
