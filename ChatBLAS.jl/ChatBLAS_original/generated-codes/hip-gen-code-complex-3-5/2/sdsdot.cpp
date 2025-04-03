#include "chatblas_hip.h"

__global__ void sdsdot_kernel( int n, float b, float *x, float *y, float *res ) {
    __shared__ float partial_sum[256];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int local_idx = threadIdx.x;
    float temp_sum = 0.0f;

    while (tid < n) {
        temp_sum += (float)x[tid] * (float)y[tid];
        tid += gridDim.x * blockDim.x;
    }

    partial_sum[local_idx] = temp_sum;
    __syncthreads();

    int idx = blockDim.x/2;
    while (idx != 0) {
        if (local_idx < idx) {
            partial_sum[local_idx] += partial_sum[local_idx + idx];
        }
        __syncthreads();
        idx /= 2;
    }

    if (local_idx == 0) {
        res[blockIdx.x] = (double)partial_sum[0] + b;
    }
}

float chatblas_sdsdot( int n, float b, float *x, float *y) {
    float *d_x, *d_y, *d_res;
    float *h_res = (float*)malloc(sizeof(float)*1);

    hipMalloc(&d_x, n * sizeof(float));
    hipMalloc(&d_y, n * sizeof(float));
    hipMalloc(&d_res, sizeof(float)*1);

    hipMemcpy(d_x, x, n*sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_y, y, n*sizeof(float), hipMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    sdsdot_kernel<<<numBlocks, blockSize>>>(n, b, d_x, d_y, d_res);

    hipMemcpy(h_res, d_res, sizeof(float)*1, hipMemcpyDeviceToHost);

    float result = h_res[0];

    hipFree(d_x);
    hipFree(d_y);
    hipFree(d_res);
    free(h_res);

    return result;
}