#include "chatblas_hip.h"

__global__ void sscal_kernel( int n, float a, float *x )
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n)
        x[i] = a * x[i];
}

void chatblas_sscal( int n, float a, float *x)
{
    float *dx;
    int size = n * sizeof(float);

    // allocate device memory
    hipMalloc((void**)&dx, size);

    // copy x from host to device
    hipMemcpy(dx, x, size, hipMemcpyHostToDevice);

    // calculate grid and block sizes
    dim3 blockSize(256, 1, 1);
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x, 1, 1);

    // call the kernel
    hipLaunchKernelGGL(sscal_kernel, gridSize, blockSize, 0, 0, n, a, dx);

    // copy result back to host
    hipMemcpy(x, dx, size, hipMemcpyDeviceToHost);

    // free device memory
    hipFree(dx);
}
