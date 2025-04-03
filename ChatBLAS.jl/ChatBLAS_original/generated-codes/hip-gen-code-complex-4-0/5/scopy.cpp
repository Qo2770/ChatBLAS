#include "chatblas_hip.h"

__global__ void scopy_kernel( int n, float *x, float *y ) 
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (index < n) 
    {
        y[index] = x[index];
    }
}

void chatblas_scopy(int n, float *x, float *y) 
{
    float *x_d, *y_d;
    size_t size = n * sizeof(float);

    hipMalloc((void **)&x_d, size);
    hipMalloc((void **)&y_d, size);

    hipMemcpy(x_d, x, size, hipMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;

    hipLaunchKernelGGL(scopy_kernel, dim3(numBlocks), dim3(threadsPerBlock), 0, 0, n, x_d, y_d);

    hipMemcpy(y, y_d, size, hipMemcpyDeviceToHost);

    hipFree(x_d);
    hipFree(y_d);
}