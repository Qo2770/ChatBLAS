#include "chatblas_hip.h"

__global__ void snrm2_kernel( int n, float *x, float *res) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        float val = x[idx];
        atomicAdd(res, val * val);
    }
}

float chatblas_snrm2(int n, float *x) {
    float *d_x, *d_res;
    float h_res;

    hipMalloc(&d_x, n * sizeof(float));
    hipMalloc(&d_res, sizeof(float));

    hipMemcpy(d_x, x, n * sizeof(float), hipMemcpyHostToDevice);
    hipMemset(d_res, 0, sizeof(float));

    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;

    snrm2_kernel<<<grid_size, block_size>>>(n, d_x, d_res);

    hipMemcpy(&h_res, d_res, sizeof(float), hipMemcpyDeviceToHost);

    hipFree(d_x);
    hipFree(d_res);

    return sqrt(h_res);
}