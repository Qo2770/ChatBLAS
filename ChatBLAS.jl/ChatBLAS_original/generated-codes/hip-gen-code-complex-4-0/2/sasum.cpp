#include "chatblas_hip.h"

__global__ void sasum_kernel(int n, float *x, float *sum) {
    extern __shared__ float shm[];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    shm[threadIdx.x] = (idx < n) ? abs(x[idx]) : 0;
    
    __syncthreads();
    
    for(int stride = blockDim.x / 2; stride> 0; stride /= 2) {
        if(threadIdx.x < stride) {
            shm[threadIdx.x] += shm[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    if(threadIdx.x == 0) atomicAdd(sum, shm[0]);
}

float chatblas_sasum(int n, float *x) {
    float sum_h = 0, *sum_d, *x_d;
    int size = n * sizeof(float);

    hipMalloc((void**)&sum_d, sizeof(float));
    hipMalloc((void**)&x_d, size);
    
    hipMemcpy(x_d, x, size, hipMemcpyHostToDevice);
    hipMemset(sum_d, 0, sizeof(float));
    
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    blockDim = blockSize;
    gridDim = gridSize;

    sasum_kernel<<<gridDim, blockDim, blockSize * sizeof(float)>>>(n, x_d, sum_d);

    hipMemcpy(&sum_h, sum_d, sizeof(float), hipMemcpyDeviceToHost);

    hipFree(sum_d);
    hipFree(x_d);
    
    return sum_h;
}