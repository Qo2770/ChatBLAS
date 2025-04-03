#include "chatblas_hip.h"

__global__ void scopy_kernel(int n, float *x, float *y) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (index < n) {
        y[index] = x[index];
    }
}

void chatblas_scopy(int n, float *x, float *y) {
    float *d_x, *d_y;
    
    hipMalloc(&d_x, n * sizeof(float));
    hipMalloc(&d_y, n * sizeof(float));
    
    hipMemcpy(d_x, x, n * sizeof(float), hipMemcpyHostToDevice);
    
    dim3 block_size(256);
    dim3 grid_size((n + block_size.x - 1) / block_size.x);
    
    scopy_kernel<<<grid_size, block_size>>>(n, d_x, d_y);
    
    hipMemcpy(y, d_y, n * sizeof(float), hipMemcpyDeviceToHost);
    
    hipFree(d_x);
    hipFree(d_y);
}