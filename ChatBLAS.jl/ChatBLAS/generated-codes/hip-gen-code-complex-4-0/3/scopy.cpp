#include "chatblas_hip.h"

__global__ void scopy_kernel( int n, float *x, float *y ) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < n)
        y[i] = x[i];
}

void chatblas_scopy(int n, float *x, float *y) {

    float *x_gpu, *y_gpu;
    int nBlocks = (n + 255) / 256;
    hipMalloc((void**)&x_gpu, n * sizeof(float));
    hipMalloc((void**)&y_gpu, n * sizeof(float));

    hipMemcpy(x_gpu, x, n * sizeof(float), hipMemcpyHostToDevice);
    
    hipLaunchKernelGGL(scopy_kernel, dim3(nBlocks), dim3(256), 0, 0, n, x_gpu, y_gpu);

    hipMemcpy(y, y_gpu, n * sizeof(float), hipMemcpyDeviceToHost);

    hipFree(x_gpu);
    hipFree(y_gpu);
}