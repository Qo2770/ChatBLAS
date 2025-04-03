#include "chatblas_hip.h"

__global__ void sdsdot_kernel( int n, float b, float *x, float *y, float *res ) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < n){
        atomicAdd(res, (double)x[tid] * (double)y[tid]);
    }
}

float chatblas_sdsdot( int n, float b, float *x, float *y) {
    float *d_x, *d_y, *d_res;
    float h_res = 0.0;

    hipMalloc(&d_x, n * sizeof(float));
    hipMalloc(&d_y, n * sizeof(float));
    hipMemcpy(d_x, x, n * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_y, y, n * sizeof(float), hipMemcpyHostToDevice);

    hipMalloc(&d_res, sizeof(float));
    hipMemcpy(d_res, &h_res, sizeof(float), hipMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    sdsdot_kernel<<<gridSize, blockSize>>>(n, b, d_x, d_y, d_res);
    
    hipMemcpy(&h_res, d_res, sizeof(float), hipMemcpyDeviceToHost);

    hipFree(d_x);
    hipFree(d_y);
    hipFree(d_res);

    return h_res;
}