#include "chatblas_cuda.h"

__global__ void sasum_kernel(int n, float *x, float *result) {
    extern __shared__ float shared_data[];

    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + tid;
    float temp = 0.0;

    if (index < n) {
        temp = fabsf(x[index]);
    }

    shared_data[tid] = temp;
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
    float *d_x, *d_result, h_result = 0.0;
    cudaMalloc((void**)&d_x, n * sizeof(float));
    cudaMalloc((void**)&d_result, sizeof(float));
    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_result, &h_result, sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    sasum_kernel<<<gridSize, blockSize, blockSize * sizeof(float)>>>(n, d_x, d_result);

    cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_result);

    return h_result;
}
