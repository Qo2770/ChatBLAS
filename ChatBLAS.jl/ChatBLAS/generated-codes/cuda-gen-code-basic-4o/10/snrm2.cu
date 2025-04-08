#include "chatblas_cuda.h"

__global__ void computeSquaredNorm(int n, float *x, float *result) {
    extern __shared__ float sharedData[];
    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + tid;

    // Load elements into shared memory and compute squared values
    sharedData[tid] = (index < n) ? x[index] * x[index] : 0.0f;
    __syncthreads();

    // Reduce within each block
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sharedData[tid] += sharedData[tid + stride];
        }
        __syncthreads();
    }

    // Write result of this block to global memory
    if (tid == 0) {
        atomicAdd(result, sharedData[0]);
    }
}

float chatblas_snrm2(int n, float *x) {
    float *d_x, *d_result, h_result;
    cudaMalloc((void **)&d_x, n * sizeof(float));
    cudaMalloc((void **)&d_result, sizeof(float));
    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_result, 0, sizeof(float));

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    computeSquaredNorm<<<numBlocks, blockSize, blockSize * sizeof(float)>>>(n, d_x, d_result);
    cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_result);

    return sqrtf(h_result);
}
