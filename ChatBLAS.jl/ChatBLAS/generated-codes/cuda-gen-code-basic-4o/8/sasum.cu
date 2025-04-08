#include "chatblas_cuda.h"

__global__ void compute_absolute_sum(int n, float *x, float *result) {
    extern __shared__ float shared_data[];
  
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    float temp = 0.0;

    if (i < n) {
        temp = fabsf(x[i]);
    }

    shared_data[tid] = temp;
    __syncthreads();

    // Reduce the sums within the block
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        __syncthreads();
    }

    // Atomically add the result for this block to the global result
    if (tid == 0) {
        atomicAdd(result, shared_data[0]);
    }
}

float chatblas_sasum(int n, float *x) {
    float *d_x, *d_result;
    float h_result = 0.0f;

    cudaMalloc((void **)&d_x, n * sizeof(float));
    cudaMalloc((void **)&d_result, sizeof(float));
    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_result, &h_result, sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    compute_absolute_sum<<<gridSize, blockSize, blockSize * sizeof(float)>>>(n, d_x, d_result);

    cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_result);

    return h_result;
}
