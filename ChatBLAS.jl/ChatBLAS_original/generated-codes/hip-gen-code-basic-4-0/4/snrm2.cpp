#include "chatblas_hip.h"

__global__ void snrm2_kernel(int n, float* x, float* res)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    float temp;

    if (idx < n)
    {
        temp = x[idx];
        atomicAdd(res, temp*temp);
    }
}

float chatblas_snrm2(int n, float* x)
{
    float* d_x, *d_res;
    float* h_res = (float*)malloc(sizeof(float));
    float result;
    int size = n * sizeof(float);

    // Allocate device memory
    hipMalloc((void**)&d_x, size);
    hipMalloc((void**)&d_res, sizeof(float));
    
    // Transfer data from host to device
    hipMemcpy(d_x, x, size, hipMemcpyHostToDevice);

    // Launch Kernel
    hipLaunchKernelGGL(snrm2_kernel, dim3((n + 255) / 256), dim3(256), 0, 0, n, d_x, d_res);
    
    // Transfer data from device to host
    hipMemcpy(h_res, d_res, sizeof(float), hipMemcpyDeviceToHost);

    // Calculate Euclidean norm
    result = sqrt(*h_res);

    // Free device memory
    hipFree(d_x);
    hipFree(d_res);

    return result;
}
