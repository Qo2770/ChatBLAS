#include "chatblas_cuda.h"

__global__ void sdot_kernel(int n, float *x, float *y, float *res) {
    extern __shared__ float shared_dot[];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x;
    
    float localSum = 0.0f;
    while (tid < n) {
        localSum += x[tid] * y[tid];
        tid += blockDim.x * gridDim.x;
    }
    shared_dot[lane] = localSum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (lane < stride) {
            shared_dot[lane] += shared_dot[lane + stride];
        }
        __syncthreads();
    }

    if (lane == 0) {
        atomicAdd(res, shared_dot[0]);
    }
}

float chatblas_sdot(int n, float *x, float *y) {
    float *d_x, *d_y, *d_res;
    float h_res = 0.0f;

    cudaMalloc((void**)&d_x, n * sizeof(float));
    cudaMalloc((void**)&d_y, n * sizeof(float));
    cudaMalloc((void**)&d_res, sizeof(float));

    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_res, &h_res, sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    sdot_kernel<<<numBlocks, blockSize, blockSize * sizeof(float)>>>(n, d_x, d_y, d_res);

    cudaMemcpy(&h_res, d_res, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_res);

    return h_res;
}