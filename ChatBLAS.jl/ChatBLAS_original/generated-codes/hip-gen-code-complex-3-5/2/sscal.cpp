#include "chatblas_hip.h"

__global__ void sscal_kernel( int n, float a , float *x ) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < n) {
        x[tid] *= a;
    }
}

void chatblas_sscal( int n, float a, float *x) {
    float *d_x;
    hipMalloc(&d_x, n * sizeof(float));
    hipMemcpy(d_x, x, n * sizeof(float), hipMemcpyHostToDevice);

    int block_size = 256; 
    int num_blocks = (n + block_size - 1) / block_size;

    sscal_kernel<<<num_blocks, block_size>>>(n, a, d_x);

    hipMemcpy(x, d_x, n * sizeof(float), hipMemcpyDeviceToHost);
    hipFree(d_x);
}