#include "chatblas_hip.h"

__global__ void sdsdot_kernel( int n, float b, float *x, float *y, float *res ) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int block_size = blockDim.x;
    int idx = blockIdx.x * block_size + tid;

    float acc = 0.0f;
    while (idx < n) {
        sdata[tid] = (float)x[idx] * (float)y[idx];
        __syncthreads();

        for (int offset = block_size / 2; offset > 0; offset >>= 1) {
            if (tid < offset) {
                sdata[tid] += sdata[tid + offset];
            }
            __syncthreads();
        }

        if (tid == 0) {
            atomicAdd(&acc, sdata[0]);
        }

        idx += gridDim.x * block_size;
    }

    if (tid == 0) {
        atomicAdd(res, acc + b);
    }
}

float chatblas_sdsdot( int n, float b, float *x, float *y) {
    float *d_x, *d_y, *d_res;
    float h_res = 0.0f;

    hipMalloc(&d_x, n * sizeof(float));
    hipMalloc(&d_y, n * sizeof(float));
    hipMalloc(&d_res, sizeof(float));

    hipMemcpy(d_x, x, n * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_y, y, n * sizeof(float), hipMemcpyHostToDevice);

    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;

    sdsdot_kernel<<<grid_size, block_size, block_size * sizeof(float)>>>(n, b, d_x, d_y, d_res);

    hipMemcpy(&h_res, d_res, sizeof(float), hipMemcpyDeviceToHost);

    hipFree(d_x);
    hipFree(d_y);
    hipFree(d_res);

    return h_res;
}