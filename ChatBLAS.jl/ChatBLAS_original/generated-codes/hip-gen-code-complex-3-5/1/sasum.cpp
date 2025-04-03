#include "chatblas_hip.h"
__global__ void sasum_kernel(int n, float *x, float *sum) {
    __shared__ float sdata[256];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (i < n) ? fabs(x[i]) : 0;
    __syncthreads();
    
    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        *sum = sdata[0];
    }
}

float chatblas_sasum(int n, float *x) {
    float *d_x, *d_sum, h_sum;
    int block_size = 256;
    int num_blocks = (n + block_size - 1) / block_size;
    
    hipMalloc(&d_x, sizeof(float) * n);
    hipMalloc(&d_sum, sizeof(float));
    
    hipMemcpy(d_x, x, sizeof(float) * n, hipMemcpyHostToDevice);
    
    sasum_kernel<<<num_blocks, block_size>>>(n, d_x, d_sum);
    
    hipMemcpy(&h_sum, d_sum, sizeof(float), hipMemcpyDeviceToHost);
    
    hipFree(d_x);
    hipFree(d_sum);
    
    return h_sum;
}
