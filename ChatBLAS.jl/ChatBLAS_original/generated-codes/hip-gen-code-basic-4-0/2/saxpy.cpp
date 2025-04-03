#include "chatblas_hip.h"

__global__ void saxpy_kernel(int n, float a, float *x, float *y) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) 
      y[i] = a * x[i] + y[i];
}

void chatblas_saxpy(int n, float a, float *x, float *y) 
{
    float *d_x, *d_y;

    // Allocate device memory for x and y vectors
    hipMalloc((void **)&d_x, n * sizeof(float));
    hipMalloc((void **)&d_y, n * sizeof(float));

    // Copy vectors x and y to device memory
    hipMemcpy(d_x, x, n * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_y, y, n * sizeof(float), hipMemcpyHostToDevice);

    // Execute the kernel
    saxpy_kernel<<<(n+255)/256, 256>>>(n, a, d_x, d_y);

    // Copy the result back to host
    hipMemcpy(y, d_y, n * sizeof(float), hipMemcpyDeviceToHost);

    // Free device memory
    hipFree(d_x);
    hipFree(d_y);
}
