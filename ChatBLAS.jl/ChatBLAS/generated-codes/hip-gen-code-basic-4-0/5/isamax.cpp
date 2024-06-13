#include "chatblas_hip.h"

__global__ void isamax_kernel(int n, float *x, int *ind) 
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    __shared__ float maxval;
    __shared__ int maxindex;

    if(i == 0) 
    {
        maxval = fabsf(x[0]);
        maxindex = 0;
    }

    __syncthreads();

    if(i < n) 
    {
        float val = fabsf(x[i]);

        if(val > maxval) 
        {
            maxval = val;
            maxindex = i;
        }
    }

    __syncthreads();

    if(i == 0)
        *ind = maxindex;
}

int chatblas_isamax(int n, float *x) 
{
    float *d_x;
    int *d_ind;

    hipMalloc(&d_x, n * sizeof(float));
    hipMalloc(&d_ind, sizeof(int));

    hipMemcpy(d_x, x, n * sizeof(float), hipMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    hipLaunchKernelGGL(isamax_kernel, dim3(gridSize), dim3(blockSize), 0, 0, n, d_x, d_ind);

    int ind;
    hipMemcpy(&ind, d_ind, sizeof(int), hipMemcpyDeviceToHost);

    hipFree(d_x);
    hipFree(d_ind);

    return ind;
}
