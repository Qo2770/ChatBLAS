#include <cuda_runtime.h>
#include "chatblas_cuda.h"

// CUDA kernel for scaling a vector
__global__ void sscal_kernel(int n, float a, float *x) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        x[i] *= a;
    }
}

// Host function
void chatblas_sscal(int n, float a, float *x) {
    float *d_x;
    size_t size = n * sizeof(float);
    
    // Allocate device memory
    cudaMalloc((void**)&d_x, size);

    // Copy vector x from host to device
    cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice);

    // Define block and grid size
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    // Launch CUDA kernel
    sscal_kernel<<<blocksPerGrid, threadsPerBlock>>>(n, a, d_x);

    // Copy result back to host
    cudaMemcpy(x, d_x, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_x);
}
