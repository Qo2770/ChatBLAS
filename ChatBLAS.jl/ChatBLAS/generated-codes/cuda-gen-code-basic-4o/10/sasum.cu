#include "chatblas_cuda.h"

__global__ void sasum_kernel(float *x, float *result, int n) {
    extern __shared__ float shared_data[];
    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + tid;
    shared_data[tid] = (index < n) ? fabsf(x[index]) : 0.0f;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(result, shared_data[0]);
    }
}

float chatblas_sasum(int n, float *x) {
    float *d_x, *d_result;
    float result = 0.0f;

    cudaMalloc((void **)&d_x, n * sizeof(float));
    cudaMalloc((void **)&d_result, sizeof(float));
    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_result, &result, sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    sasum_kernel<<<gridSize, blockSize, blockSize * sizeof(float)>>>(d_x, d_result, n);

    cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_result);

    return result;
}
