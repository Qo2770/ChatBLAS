#include "chatblas_cuda.h"

// CUDA Kernel function to copy the elements of vector x into vector y
__global__ void copy_kernel(int n, float *x, float *y) 
{
    // Calculate the index for each thread
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    
    // Ensure we don't go out of bounds
    if (index < n) 
    {
        y[index] = x[index];
    }
}

void chatblas_scopy(int n, float *x, float *y) 
{
    // Allocate device memory
    float* d_x;
    float* d_y;
    
    cudaMalloc((void**)&d_x, n * sizeof(float));
    cudaMalloc((void**)&d_y, n * sizeof(float));

    // Copy vectors from host memory to device memory
    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
    
    // Calculate number of blocks required, assuming a fixed number of threads
    int blocks = (n + 255) / 256;

    // Launch copy kernel on GPU
    copy_kernel<<<blocks, 256>>>(n, d_x, d_y);
    
    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();
    
    // Copy the array 'd_y' back from the GPU to the CPU
    cudaMemcpy(y, d_y, n * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(d_x);
    cudaFree(d_y);
}
