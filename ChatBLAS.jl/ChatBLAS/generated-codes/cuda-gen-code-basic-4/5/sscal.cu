#include "chatblas_cuda.h"
#include<math.h>

// CUDA kernel for vectors scaling 
__global__ void scaleKernel(int n, float a, float *x)
{
    int index = threadIdx.x;
    int stride = blockDim.x;

    for (int i = index; i < n; i += stride)
        x[i] = a * x[i];
}

// C function to scale vectors using CUDA
void chatblas_sscal(int n, float a, float *x)
{
    // Get available device 
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if(deviceCount == 0) {
        fprintf(stderr, "error: no devices supporting CUDA.\n");
        exit(EXIT_FAILURE);
    }
    int dev = 0;
    cudaSetDevice(dev);

    // Allocate and copy to the device
    float* d_x;
    cudaMalloc((void**)&d_x, n*sizeof(float));
    cudaMemcpy(d_x, x, n*sizeof(float), cudaMemcpyHostToDevice);

    // Launch scale kernel on the device
    int threadsPerBlock = 256;
    int blocksPerGrid =(n + threadsPerBlock - 1) / threadsPerBlock;
    scaleKernel<<<blocksPerGrid, threadsPerBlock>>>(n, a, d_x);

    // Copy result back to host
    cudaMemcpy(x, d_x, n*sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_x);
}
