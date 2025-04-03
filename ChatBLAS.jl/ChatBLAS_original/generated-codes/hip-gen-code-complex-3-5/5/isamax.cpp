#include "chatblas_hip.h"

__global__ void isamax_kernel(int n, float *x, int *ind) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n) {
        atomicMax((unsigned int*)ind, abs(x[tid]) > abs(x[*ind]) ? tid : *ind);
    }
}

int chatblas_isamax(int n, float *x) {
    float *d_x;
    int *d_ind;
    int h_ind;
    
    hipMalloc(&d_x, n * sizeof(float));
    hipMalloc(&d_ind, sizeof(int));
    
    hipMemcpy(d_x, x, n * sizeof(float), hipMemcpyHostToDevice);
    hipMemset(d_ind, 0, sizeof(int));
    
    int block_size = 256;
    int num_blocks = (n + block_size - 1) / block_size;
    
    isamax_kernel<<<num_blocks, block_size>>>(n, d_x, d_ind);
    
    hipMemcpy(&h_ind, d_ind, sizeof(int), hipMemcpyDeviceToHost);
    
    hipFree(d_x);
    hipFree(d_ind);
    
    return h_ind;
}