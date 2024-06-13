#include "chatblas_hip.h"
#include "hip/hip_runtime.h"

__global__ void sdsdot_kernel( int n, float b, float *x, float *y, float *res ) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        res[idx] = x[idx] * y[idx];
    }
    __syncthreads();
    if (idx == 0) {
        for (int i = 1; i < n; i++) {
            res[0] += res[i];
        }
    res[0] += b;
    }
}

extern "C"
float chatblas_sdsdot( int n, float b, float *x, float *y) {
    float *x_gpu, *y_gpu, *res_gpu;
    float res;
    size_t size = n * sizeof(float);

    hipMalloc((void**)&x_gpu, size);
    hipMalloc((void**)&y_gpu, size);
    hipMalloc((void**)&res_gpu, size);

    hipMemcpy(x_gpu, x, size, hipMemcpyHostToDevice);
    hipMemcpy(y_gpu, y, size, hipMemcpyHostToDevice);
  
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    hipLaunchKernelGGL(sdsdot_kernel, dim3(blocksPerGrid), dim3(threadsPerBlock), 0, 0, n, b, x_gpu, y_gpu, res_gpu);

    hipMemcpy(&res, res_gpu, sizeof(float), hipMemcpyDeviceToHost);

    hipFree(x_gpu);
    hipFree(y_gpu);
    hipFree(res_gpu);

    return (double)res;
}
