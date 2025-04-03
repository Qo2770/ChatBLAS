#include "chatblas_hip.h"

__global__ void sasum_kernel(int n, float *x, float *sum) {
    __shared__ float partialSum[256];
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    int tid = threadIdx.x;

    if (id < n) {
        partialSum[tid] = fabsf(x[id]);
    } else {
        partialSum[tid] = 0.0f;
    }
    __syncthreads();

    for (int d = 1; d < blockDim.x; d *= 2) {
        if (tid % (2 * d) == 0) {
            partialSum[tid] += partialSum[tid + d];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(sum, partialSum[0]);
    }
}

float chatblas_sasum(int n, float *x) {
    float *d_x, *d_sum;
    float h_sum;

    hipMalloc(&d_x, n * sizeof(float));
    hipMemcpy(d_x, x, n * sizeof(float), hipMemcpyHostToDevice);
    hipMalloc(&d_sum, sizeof(float));

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    sasum_kernel<<<numBlocks, blockSize>>>(n, d_x, d_sum);
    
    hipMemcpy(&h_sum, d_sum, sizeof(float), hipMemcpyDeviceToHost);

    hipFree(d_x);
    hipFree(d_sum);

    return h_sum;
}