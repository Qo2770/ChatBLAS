#include "chatblas_cuda.h"

__global__ void dotProductKernel(float *x, float *y, float *out, int n) {
    __shared__ float buffer[256];
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    float tempSum = 0;

    // Boundary check
    if (idx < n) {
        tempSum += x[idx] * y[idx];
    }

    buffer[threadIdx.x] = tempSum;
    __syncthreads();
  
    // Reduction
    for (int i = blockDim.x / 2; i > 0; i >>= 1) {
        if (threadIdx.x < i)
            buffer[threadIdx.x] += buffer[threadIdx.x + i];
        __syncthreads();
    }  

    if (threadIdx.x == 0)
        atomicAdd(out, buffer[0]);
}

float chatblas_sdot(int n, float *x, float *y) {
    float *dev_x, *dev_y, *dev_out;
    float result;
    float *out = (float*)malloc(sizeof(float));

    // Allocate device memory
    cudaMalloc((void**)&dev_x, n * sizeof(float));
    cudaMalloc((void**)&dev_y, n * sizeof(float));
    cudaMalloc((void**)&dev_out, sizeof(float));

    // Copy x and y vectors to device memory
    cudaMemcpy(dev_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_y, y, n * sizeof(float), cudaMemcpyHostToDevice);

    // Set output to 0
    cudaMemset(dev_out, 0, sizeof(float));

    // Launch dotProductKernel
    dotProductKernel<<<(n + 255) / 256, 256>>>(dev_x, dev_y, dev_out, n);
 
    // Copy result back to host memory
    cudaMemcpy(out, dev_out, sizeof(float), cudaMemcpyDeviceToHost);

    // Cleanup device memory
    cudaFree(dev_x);
    cudaFree(dev_y);
    cudaFree(dev_out);

    result = out[0];

    free(out);

    return result;
}
