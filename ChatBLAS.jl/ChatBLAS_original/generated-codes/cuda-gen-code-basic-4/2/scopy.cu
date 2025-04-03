#include "chatblas_cuda.h"

// Kernel function to copy contents of one array to another
__global__ void copy_kernel(int n, float *x, float *y)
{
    // Get the thread ID which equals the vector index
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n) {
        y[i] = x[i];
    }
}

void chatblas_scopy(int n, float *x, float *y) 
{
   int blockSize = 1024;
   int numBlocks = (n + blockSize - 1) / blockSize;

   // Allocate device memory
   float *d_x, *d_y;
   cudaMalloc((void **)&d_x, n * sizeof(float));
   cudaMalloc((void **)&d_y, n * sizeof(float));

   // Copy vector x from host to device memory
   cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);

   //call kernel function 
   copy_kernel<<<numBlocks, blockSize>>>(n, d_x, d_y);

   // Copy the result back to host memory
   cudaMemcpy(y, d_y, n * sizeof(float), cudaMemcpyDeviceToHost);

   // Free device memory
   cudaFree(d_x); cudaFree(d_y);
}
