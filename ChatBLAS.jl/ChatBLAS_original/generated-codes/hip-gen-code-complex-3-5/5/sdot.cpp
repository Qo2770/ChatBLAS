#include "chatblas_hip.h"
__global__ void sdot_kernel(int n, float *x, float *y, float *res) {
    __shared__ float shared_mem[256];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    shared_mem[tid] = (i < n) ? x[i] * y[i] : 0;
    __syncthreads();

    for(int s = 1; s < blockDim.x; s *= 2) {
        if(tid % (2 * s) == 0) {
            shared_mem[tid] += shared_mem[tid + s];
        }
        __syncthreads();
    }

    if(tid == 0) {
        res[blockIdx.x] = shared_mem[0];
    }
}

float chatblas_sdot( int n, float *x, float *y) {
    const int block_size = 256;
    const int num_blocks = (n + block_size - 1) / block_size;

    float *d_x, *d_y, *d_res, res;
    hipMalloc(&d_x, n * sizeof(float));
    hipMalloc(&d_y, n * sizeof(float));
    hipMalloc(&d_res, num_blocks * sizeof(float));

    hipMemcpy(d_x, x, n * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_y, y, n * sizeof(float), hipMemcpyHostToDevice);

    sdot_kernel<<<num_blocks, block_size>>>(n, d_x, d_y, d_res);

    float *h_res = new float[num_blocks];
    hipMemcpy(h_res, d_res, num_blocks * sizeof(float), hipMemcpyDeviceToHost);

    res = 0;
    for(int i = 0; i < num_blocks; i++) {
        res += h_res[i];
    }

    delete[] h_res;
    hipFree(d_x);
    hipFree(d_y);
    hipFree(d_res);

    return res;
}