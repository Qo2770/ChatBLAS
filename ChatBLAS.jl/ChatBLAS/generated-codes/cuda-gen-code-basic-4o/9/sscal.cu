#include "chatblas_cuda.h"

// CUDA kernel to scale the vector x by the constant a
__global__ void sscale_kernel(int n, float a, float *x) {
    // Calculate the global thread ID
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Ensure we do not access beyond the array
    if (index < n) {
        x[index] *= a;
    }
}

void chatblas_sscal(int n, float a, float *x) {
    // Define block size and number of blocks
    int blockSize = 256; // You can adjust this depending on your device
    int numBlocks = (n + blockSize - 1) / blockSize; // Compute number of blocks needed

    // Launch the kernel
    sscale_kernel<<<numBlocks, blockSize>>>(n, a, x);

    // Synchronize to ensure that all threads have completed
    cudaDeviceSynchronize();
}
