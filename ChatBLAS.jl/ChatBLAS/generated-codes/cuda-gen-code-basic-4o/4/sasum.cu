#include "chatblas_cuda.h"

__global__ void sasum_kernel(int n, float *x, float *result) {
    extern __shared__ float shared[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    float temp = 0.0f;

    // Each thread sums part of the array
    if (i < n) {
        temp = fabsf(x[i]);
    }

    shared[tid] = temp;
    __syncthreads();

    // Parallel reduction within block
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }

    // The first thread in the block writes the result to global memory
    if (tid == 0) {
        atomicAdd(result, shared[0]);
    }
}

float chatblas_sasum(int n, float *x) {
    float *d_x, *d_result, h_result = 0.0f;

    cudaMalloc((void **)&d_x, n * sizeof(float));
    cudaMalloc((void **)&d_result, sizeof(float));
    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_result, &h_result, sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    size_t sharedMemSize = blockSize * sizeof(float);

    sasum_kernel<<<numBlocks, blockSize, sharedMemSize>>>(n, d_x, d_result);
    cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_result);

    return h_result;
}
