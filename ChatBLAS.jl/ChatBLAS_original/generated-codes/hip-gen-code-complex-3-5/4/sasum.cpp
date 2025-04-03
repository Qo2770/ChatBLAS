#include "chatblas_hip.h"
__global__ void sasum_kernel(int n, float *x, float *sum) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    
    sdata[tid] = (i < n) ? fabsf(x[i]) : 0;
    __syncthreads();

    for(int s=1; s < blockDim.x; s *= 2) {
        if(tid % (2*s) == 0 && tid + s < blockDim.x) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if(tid == 0) atomicAdd(sum, sdata[0]);
}

float chatblas_sasum(int n, float *x) {
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    float *d_x, *d_sum;

    hipMalloc(&d_x, n * sizeof(float));
    hipMalloc(&d_sum, sizeof(float));
    hipMemcpy(d_x, x, n * sizeof(float), hipMemcpyHostToDevice);

    sasum_kernel<<<gridSize, blockSize, blockSize * sizeof(float)>>>(n, d_x, d_sum);

    float h_sum;
    hipMemcpy(&h_sum, d_sum, sizeof(float), hipMemcpyDeviceToHost);

    hipFree(d_x);
    hipFree(d_sum);

    return h_sum;
}