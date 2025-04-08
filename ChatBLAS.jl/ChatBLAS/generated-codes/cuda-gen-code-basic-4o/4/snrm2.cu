#include "chatblas_cuda.h"

__global__ void computeSquaredNorm(int n, float *x, float *result) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + tid;

    // Load elements into shared memory
    sdata[tid] = (i < n) ? x[i] * x[i] : 0.0f;
    __syncthreads();

    // Perform reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write the result for this block to global memory
    if (tid == 0) {
        atomicAdd(result, sdata[0]);
    }
}

float chatblas_snrm2(int n, float *x) {
    float *d_x, *d_result;
    float h_result = 0.0f;

    cudaMalloc((void**)&d_x, n * sizeof(float));
    cudaMalloc((void**)&d_result, sizeof(float));

    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_result, &h_result, sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    computeSquaredNorm<<<numBlocks, blockSize, blockSize * sizeof(float)>>>(n, d_x, d_result);

    cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_result);

    return sqrtf(h_result);
}
