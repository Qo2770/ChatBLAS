#include "chatblas_cuda.h"

__global__ void sasum_kernel(int n, float *x, float *sum) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n) {
        atomicAdd(sum, fabs(x[tid]));
    }
}

float chatblas_sasum(int n, float *x) {
    float *d_x, *d_sum, sum;
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
  
    cudaMalloc((void**)&d_x, n * sizeof(float));
    cudaMalloc((void**)&d_sum, sizeof(float));
  
    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_sum, 0, sizeof(float));
  
    sasum_kernel<<<grid_size, block_size>>>(n, d_x, d_sum);
  
    cudaMemcpy(&sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);
  
    cudaFree(d_x);
    cudaFree(d_sum);
  
    return sum;
}
