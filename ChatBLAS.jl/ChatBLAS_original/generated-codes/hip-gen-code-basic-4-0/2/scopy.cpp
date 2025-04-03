#include "chatblas_hip.h"

__global__ void scopy_kernel( int n, float *x, float *y ) {
   int tid = blockDim.x * blockIdx.x + threadIdx.x;
   if (tid < n) y[tid] = x[tid];
}

void chatblas_scopy(int n, float *x, float *y) {
    float* x_gpu;
    float* y_gpu;
    int size = n * sizeof(float);
    
    hipMallocManaged(&x_gpu, size);
    hipMallocManaged(&y_gpu, size);
    
    hipMemcpy(x_gpu, x, size, hipMemcpyHostToDevice);
    
    dim3 blockSize(256,1,1);
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x, 1, 1);
    hipLaunchKernelGGL(scopy_kernel, gridSize, blockSize, 0, 0, n, x_gpu, y_gpu);
    hipDeviceSynchronize();
    
    hipMemcpy(y, y_gpu, size, hipMemcpyDeviceToHost);
    
    hipFree(x_gpu);
    hipFree(y_gpu);
}
