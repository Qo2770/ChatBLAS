#include "chatblas_hip.h"

__global__ void sscal_kernel( int n, float a , float *x ) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if( i < n ) {
        x[i] *= a;
    }
}

void chatblas_sscal( int n, float a, float *x) {
    float *d_x;
    hipMalloc((void **)&d_x, n * sizeof(float));  
    hipMemcpy(d_x, x, n * sizeof(float), hipMemcpyHostToDevice);

    const int blockSize = 256;
    const int numBlocks = (n + blockSize - 1) / blockSize;
    sscal_kernel <<< numBlocks, blockSize >>> (n, a, d_x);

    hipMemcpy(x, d_x, n * sizeof(float), hipMemcpyDeviceToHost);
    hipFree(d_x);
}