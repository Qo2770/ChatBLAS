#include "chatblas_hip.h"

__global__ void sasum_kernel(int n, float *x, float *sum) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;

    __shared__ float partialSum[256];
    float localSum = 0.0f;

    while (tid < n) {
        localSum += fabs(x[tid]);
        tid += stride;
    }

    partialSum[threadIdx.x] = localSum;

    __syncthreads();

    int i = blockDim.x / 2;
    while (i != 0) {
        if (threadIdx.x < i) {
            partialSum[threadIdx.x] += partialSum[threadIdx.x + i];
        }
        __syncthreads();
        i /= 2;
    }

    if (threadIdx.x == 0) {
        atomicAdd(sum, partialSum[0]);
    }
}

float chatblas_sasum(int n, float *x) {
    float *d_x;
    float *d_sum;
    float h_sum = 0.0f;

    hipMalloc(&d_x, n * sizeof(float));
    hipMalloc(&d_sum, sizeof(float));

    hipMemcpy(d_x, x, n * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_sum, &h_sum, sizeof(float), hipMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    sasum_kernel<<<numBlocks, blockSize>>>(n, d_x, d_sum);

    hipMemcpy(&h_sum, d_sum, sizeof(float), hipMemcpyDeviceToHost);

    hipFree(d_x);
    hipFree(d_sum);

    return h_sum;
}