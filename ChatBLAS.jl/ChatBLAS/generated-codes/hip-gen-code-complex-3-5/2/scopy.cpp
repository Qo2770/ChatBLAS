#include "chatblas_hip.h"

__global__ void scopy_kernel(int n, float *x, float *y) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        y[index] = x[index];
    }
}

void chatblas_scopy(int n, float *x, float *y) {
    float *d_x, *d_y;
    hipMalloc((void**)&d_x, n * sizeof(float));
    hipMalloc((void**)&d_y, n * sizeof(float));
    
    hipMemcpy(d_x, x, n * sizeof(float), hipMemcpyHostToDevice);

    int block_size = 256;
    int num_blocks = (n + block_size - 1) / block_size;

    scopy_kernel<<<num_blocks, block_size>>>(n, d_x, d_y);
    
    hipMemcpy(y, d_y, n * sizeof(float), hipMemcpyDeviceToHost);
    
    hipFree(d_x);
    hipFree(d_y);
}