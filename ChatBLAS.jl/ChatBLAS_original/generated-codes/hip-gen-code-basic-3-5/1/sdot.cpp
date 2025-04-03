#include "hip/hip_runtime.h"

__global__ void sdot_kernel(int n, float *x, float *y, float *res) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    __shared__ float sdata[256]; //max threads per block
    sdata[tid] = 0;

    while(tid < n) {
        sdata[tid] = x[tid] * y[tid];
        tid += blockDim.x * gridDim.x;
    }

    __syncthreads();

    for(int s = blockDim.x / 2; s > 0; s >>= 1) {
        if(tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if(threadIdx.x == 0) {
        atomicAdd(res, sdata[0]);
    }
}

float chatblas_sdot(int n, float *x, float *y) {
    float *d_x, *d_y, *d_res;
    float h_res;

    hipMalloc(&d_x, n * sizeof(float));
    hipMalloc(&d_y, n * sizeof(float));
    hipMalloc(&d_res, sizeof(float));

    hipMemcpy(d_x, x, n * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_y, y, n * sizeof(float), hipMemcpyHostToDevice);

    sdot_kernel<<<(n + 255)/256, 256>>>(n, d_x, d_y, d_res);

    hipMemcpy(&h_res, d_res, sizeof(float), hipMemcpyDeviceToHost);

    hipFree(d_x);
    hipFree(d_y);
    hipFree(d_res);

    return h_res;
}
