#include "chatblas_cuda.h"

__global__ void dotProdKernel(int n, float *x, float *y, float *result) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    __shared__ float temp[256];
    temp[threadIdx.x] = 0;

    if(index < n) {
        temp[threadIdx.x] = x[index] * y[index];
    }
    __syncthreads();

    if(threadIdx.x == 0) {
        float sum = 0;
        for(int i=0; i<256; i++) {
            sum += temp[i];
        }
        atomicAdd(result, sum);
    }
}

float chatblas_sdot(int n, float *x, float *y) {
    float *d_x, *d_y, *d_result;
    float result = 0;
    int size = n * sizeof(float); 

    cudaMalloc(&d_x, size);
    cudaMalloc(&d_y, size);
    cudaMalloc(&d_result, sizeof(float));

    cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, size, cudaMemcpyHostToDevice);

    dotProdKernel<<<(n+255)/256, 256>>>(n, d_x, d_y, d_result);

    cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost); // Result copied from device to host

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_result);
    
    return result;
}
