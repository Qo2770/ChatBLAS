#include "chatblas_hip.h"

__global__ void scopy_kernel( int n, float *x, float *y ) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = x[idx];
    }
}

void chatblas_scopy(int n, float *x, float *y) {
    
    float *d_x, *d_y;
    size_t size = n * sizeof(float);

    hipMalloc((void **) &d_x, size);
    hipMalloc((void **) &d_y, size);

    hipMemcpy(d_x, x, size, hipMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid =(n + threadsPerBlock - 1) / threadsPerBlock;

    hipLaunchKernelGGL(scopy_kernel, dim3(blocksPerGrid), dim3(threadsPerBlock), 0, 0, n, d_x, d_y);

    hipMemcpy(y, d_y, size, hipMemcpyDeviceToHost);

    hipFree(d_x);
    hipFree(d_y);
}