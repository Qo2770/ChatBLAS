#include "chatblas_cuda.h"

__global__ void sasum_kernel(int n, float *x, float *sum) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    float temp = 0.0;
    __shared__ float temp_sum[256];

    temp_sum[threadIdx.x] = (index < n) ? abs(x[index]) : 0;
    __syncthreads();

    if (threadIdx.x == 0) {
        for (int i = 0; i < blockDim.x; i++) {
            temp += temp_sum[i];
        }
        atomicAdd(sum,temp);
    }
}

float chatblas_sasum(int n, float *x) {
    float *xdev = NULL, *sumdev = NULL;
    float sum = 0.0;
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    cudaMalloc((void**)&xdev, n * sizeof(float));
    cudaMemcpy(xdev, x, n * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&sumdev, sizeof(float));
    cudaMemcpy(sumdev, &sum, sizeof(float), cudaMemcpyHostToDevice);

    sasum_kernel<<<numBlocks, blockSize>>>(n, xdev, sumdev);

    cudaMemcpy(&sum, sumdev, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(xdev);
    cudaFree(sumdev);

    return sum;
}