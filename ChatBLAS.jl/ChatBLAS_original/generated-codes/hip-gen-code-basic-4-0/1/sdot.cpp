#include "chatblas_hip.h"

__global__ void sdot_kernel(int n, float *x, float *y, float *res) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    __shared__ float temp[256];

    temp[threadIdx.x] = (idx < n) ? x[idx]*y[idx] : 0;
    __syncthreads();

    if (threadIdx.x == 0) {
        float sum = 0;
        for(int i=0; i<256; i++) {
            sum += temp[i];
        }
        atomicAdd(res, sum);
    }
}

float chatblas_sdot( int n, float *x, float *y) {
    float *x_device = NULL, *y_device = NULL, *res_device = NULL;
 
    // allocations for device memory 
    hipMalloc(&x_device, n*sizeof(float));
    hipMalloc(&y_device, n*sizeof(float));
    hipMalloc(&res_device, sizeof(float));
 
    // copying data to device
    hipMemcpy(x_device, x, n*sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(y_device, y, n*sizeof(float), hipMemcpyHostToDevice);
 
    float res_host = 0;
    hipMemcpy(res_device, &res_host, sizeof(float), hipMemcpyHostToDevice);
 
    // preparing for kernel execution
    int threadsPerBlock = 256;
    int blocksPerGrid =(n + threadsPerBlock - 1) / threadsPerBlock;
 
    // launching kernel
    hipLaunchKernelGGL(sdot_kernel, dim3(blocksPerGrid), dim3(threadsPerBlock), 0, 0, n, x_device, y_device, res_device);
 
    // copying result back to host
    hipMemcpy(&res_host, res_device, sizeof(float), hipMemcpyDeviceToHost);
 
    // free device memory
    hipFree(x_device);
    hipFree(y_device);
    hipFree(res_device);
 
    return res_host;
}
