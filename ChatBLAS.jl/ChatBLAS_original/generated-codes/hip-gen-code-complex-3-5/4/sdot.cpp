#include "chatblas_hip.h"

__global__ void sdot_kernel(int n, float *x, float *y, float *res) {
    __shared__ float shared_data[256];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    
    float dot = 0.0f;
    
    while (idx < n) {
        shared_data[tid] = x[idx] * y[idx];
        __syncthreads();

        for (int stride = 1; stride < blockDim.x; stride *= 2) {
            if (tid % (2 * stride) == 0) {
                shared_data[tid] += shared_data[tid + stride];
            }
            __syncthreads();
        }
        
        if (tid == 0) {
            atomicAdd(res, shared_data[0]);
        }
        
        idx += blockDim.x * gridDim.x;
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
    
    hipLaunchKernelGGL(sdot_kernel, dim3(256), dim3(256), 0, 0, n, d_x, d_y, d_res);
    
    hipMemcpy(&h_res, d_res, sizeof(float), hipMemcpyDeviceToHost);
    
    hipFree(d_x);
    hipFree(d_y);
    hipFree(d_res);
    
    return h_res;
}