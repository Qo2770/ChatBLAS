#include "chatblas_hip.h"

__global__ void sdot_kernel(int n, float *x, float *y, float *res) {
    __shared__ float temp[256];  // Assuming block size of 256
    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    float dot = 0.0f;
    
    while(index < n) {
        temp[tid] = x[index] * y[index];
        __syncthreads();
        
        for (int i = 128; i > 0; i /= 2) {
            if (tid < i) {
                temp[tid] += temp[tid + i];
            }
            __syncthreads();
        }
        
        if (tid == 0) {
            atomicAdd(res, temp[0]);
        }
        
        index += blockDim.x * gridDim.x;
    }
}

float chatblas_sdot(int n, float *x, float *y) {
    float *d_x, *d_y, *d_res;
    float h_res = 0.0f;

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