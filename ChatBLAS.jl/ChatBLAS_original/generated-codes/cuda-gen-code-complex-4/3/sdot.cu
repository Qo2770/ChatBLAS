#include "chatblas_cuda.h"

__global__ void sdot_kernel(int n, float *x, float *y, float *res) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    __shared__ float temp[256];
    temp[threadIdx.x] = 0;

    if(index < n) {
        temp[threadIdx.x] = x[index] * y[index];
    }

   __syncthreads();

   if(threadIdx.x == 0) {
        float sum = 0;
        for(int i = 0; i < 256; ++i)
            sum += temp[i];
        atomicAdd(res, sum);
    }
}

float chatblas_sdot( int n, float *x, float *y) {
    int size = n * sizeof(float);
    float result = 0, *d_x, *d_y, *d_result;

    // Allocate space for device copies of x, y, result
    cudaMalloc((void **)&d_x, size);
    cudaMalloc((void **)&d_y, size);
    cudaMalloc((void **)&d_result, sizeof(float));

    // Copy inputs to device
    cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, size, cudaMemcpyHostToDevice);

    // Launch sdot_kernel()
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    sdot_kernel<<<numBlocks,blockSize>>>(n, d_x, d_y, d_result);

    // Copy result back to host
    cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_x); cudaFree(d_y); cudaFree(d_result);

    return result;
}