#include "chatblas_cuda.h"

__global__ void scopy_kernel(int n, float *x, float *y) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n) {
        y[tid] = x[tid];
    }
}

void chatblas_scopy(int n, float *x, float *y) {
    float *d_x, *d_y;

    // Allocate GPU memory for x and y
    cudaMalloc((void**)&d_x, n * sizeof(float));
    cudaMalloc((void**)&d_y, n * sizeof(float));

    // Copy data from CPU to GPU
    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);

    // Define block size and number of blocks
    int blocksize = 256;
    int numBlocks = (n + blocksize - 1) / blocksize;

    // Launch the kernel
    scopy_kernel<<<numBlocks, blocksize>>>(n, d_x, d_y);

    // Copy data back from GPU to CPU
    cudaMemcpy(y, d_y, n * sizeof(float), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_x);
    cudaFree(d_y);
}
