#include "chatblas_cuda.h"

__global__ void sscal_kernel(int n, float a, float *x) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n) {
        x[tid] *= a;
    }
}

void chatblas_sscal(int n, float a, float *x) {
    float *d_x;
    int size = n * sizeof(float);
    
    // Allocate GPU memory for x
    cudaMalloc((void **)&d_x, size);
    
    // Copy input vector from host to GPU
    cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice);
    
    // Define the number of threads per block and number of blocks
    int threadsPerBlock = 256;
    int numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    // Launch the kernel
    sscal_kernel<<<numBlocks, threadsPerBlock>>>(n, a, d_x);
    
    // Copy the result back from GPU to host
    cudaMemcpy(x, d_x, size, cudaMemcpyDeviceToHost);
    
    // Free GPU memory
    cudaFree(d_x);
}
