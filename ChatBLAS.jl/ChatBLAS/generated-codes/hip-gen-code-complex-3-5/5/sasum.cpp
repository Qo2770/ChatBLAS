#include "chatblas_hip.h"

__global__ void sasum_kernel(int n, float *x, float *sum) {
    __shared__ float sdata[256];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < n) {
        sdata[tid] = fabsf(x[i]);
    } else {
        sdata[tid] = 0.0f;
    }
    
    __syncthreads();
    
    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        sum[blockIdx.x] = sdata[0];
    }
}

float chatblas_sasum(int n, float *x) {
    float *d_x, *d_sum;
    float *h_sum = (float *)malloc(sizeof(float));
    
    hipMalloc(&d_x, n * sizeof(float));
    hipMalloc(&d_sum, n * sizeof(float));
    
    hipMemcpy(d_x, x, n * sizeof(float), hipMemcpyHostToDevice);
    
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    
    sasum_kernel<<<numBlocks, blockSize>>>(n, d_x, d_sum);
    
    hipMemcpy(h_sum, d_sum, sizeof(float), hipMemcpyDeviceToHost);
    
    float result = *h_sum;
    
    hipFree(d_x);
    hipFree(d_sum);
    free(h_sum);
    
    return result;
}