#include "chatblas_hip.h"

__global__ void isamax_kernel(int n, float *x, int *ind) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n) {
        if (abs(x[tid]) > x[ind[0]]) {
            ind[0] = tid;
        }
    }
}

int chatblas_isamax(int n, float *x) {
    int *d_ind;
    float *d_x;
    int *h_ind = (int*)malloc(sizeof(int));
    
    hipMalloc(&d_x, n * sizeof(float));
    hipMalloc(&d_ind, sizeof(int));
    
    hipMemcpy(d_x, x, n * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_ind, h_ind, sizeof(int), hipMemcpyHostToDevice);
    
    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);
    
    isamax_kernel <<<grid, block>>> (n, d_x, d_ind);
    
    hipMemcpy(h_ind, d_ind, sizeof(int), hipMemcpyDeviceToHost);
    
    int result = h_ind[0];
    
    hipFree(d_x);
    hipFree(d_ind);
    free(h_ind);
    
    return result;
}