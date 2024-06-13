#include "chatblas_hip.h"

__global__ void sasum_kernel(int n, float *x, float *sum) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    __shared__ float partialSum[256];
    float tempSum = 0.0f;

    for (int i = tid; i < n; i += stride) {
        tempSum += abs(x[i]);
    }

    partialSum[threadIdx.x] = tempSum;

    __syncthreads();

    // Reduction to calculate final sum
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            partialSum[threadIdx.x] += partialSum[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        sum[blockIdx.x] = partialSum[0];
    }
}

float chatblas_sasum(int n, float *x) {
    float sum;
    float *d_x, *d_sum;

    hipMalloc(&d_x, n * sizeof(float));
    hipMalloc(&d_sum, sizeof(float));

    hipMemcpy(d_x, x, n * sizeof(float), hipMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    sasum_kernel<<<gridSize, blockSize>>>(n, d_x, d_sum);

    hipMemcpy(&sum, d_sum, sizeof(float), hipMemcpyDeviceToHost);

    hipFree(d_x);
    hipFree(d_sum);

    return sum;
}