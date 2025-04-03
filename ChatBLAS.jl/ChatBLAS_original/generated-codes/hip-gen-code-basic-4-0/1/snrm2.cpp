#include "chatblas_hip.h"

__global__ void snrm2_kernel(int n, float *x, float *res) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ float partialSum;
    if (i < n) {
        atomicAdd(&partialSum, x[i] * x[i]);
        __syncthreads();
        if (threadIdx.x == 0)
            atomicAdd(res, sqrtf(partialSum));
    }
}

float chatblas_snrm2(int n, float *x) {
    float *d_x, *d_res;
    float res;

    // Memory allocation on the GPU
    hipMalloc((void **) &d_x, n * sizeof(float));
    hipMalloc((void **) &d_res, sizeof(float));

    // Memory transferring from the CPU to the GPU
    hipMemcpy(d_x, x, n * sizeof(float), hipMemcpyHostToDevice);
    hipMemset(d_res, 0, sizeof(float));

    // Calling the kernel function
    dim3 block_size(256);
    dim3 grid_size((n + block_size.x - 1) / block_size.x);
    hipLaunchKernelGGL(snrm2_kernel, dim3(grid_size), dim3(block_size), 0, 0, n, d_x, d_res);

    // Memory transferring from the GPU to the CPU
    hipMemcpy(&res, d_res, sizeof(float), hipMemcpyDeviceToHost);

    // Freeing the allocated GPU memory
    hipFree(d_x);
    hipFree(d_res);

    return res;
}
