#include "chatblas_hip.h"

__global__ void sdot_kernel(int n, float *x, float *y, float *res) {
    __shared__ float cache[256]; // assuming max block size is 256
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    cache[tid] = (i < n) ? x[i] * y[i] : 0.0;
    
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            cache[tid] += cache[tid + stride];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(res, cache[0]);
    }
}

float chatblas_sdot(int n, float *x, float *y) {
    float *d_x, *d_y, *d_res;
    float h_res = 0.0;

    hipMalloc(&d_x, n * sizeof(float));
    hipMalloc(&d_y, n * sizeof(float));
    hipMalloc(&d_res, sizeof(float));

    hipMemcpy(d_x, x, n * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_y, y, n * sizeof(float), hipMemcpyHostToDevice);

    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;

    sdot_kernel<<<grid_size, block_size>>>(n, d_x, d_y, d_res);

    hipMemcpy(&h_res, d_res, sizeof(float), hipMemcpyDeviceToHost);

    hipFree(d_x);
    hipFree(d_y);
    hipFree(d_res);

    return h_res;
}