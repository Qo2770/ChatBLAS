#include "chatblas_hip.h"

__global__ void sscal_kernel(int n, float a, float *x)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(idx < n) 
        x[idx] *= a;
}

void chatblas_sscal(int n, float a, float *x)
{
    float *d_x;  //device variable declaration

    //device memory allocation
    hipMalloc((void**)&d_x, n * sizeof(float));

    //transfer from host to device
    hipMemcpy(d_x, x, n * sizeof(float), hipMemcpyHostToDevice);

    //grid and block dimensions
    dim3 grid((n-1)/1024 + 1, 1, 1);
    dim3 block(1024, 1, 1); 

    hipLaunchKernelGGL(sscal_kernel, grid, block, 0, 0, n, a, d_x);

    //transfer from device to host
    hipMemcpy(x, d_x, n * sizeof(float), hipMemcpyDeviceToHost);

    //free device memory
    hipFree(d_x);
}
