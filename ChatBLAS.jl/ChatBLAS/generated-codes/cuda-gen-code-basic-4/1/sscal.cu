#include "chatblas_cuda.h"
#include <math.h>

// CUDA kernel to scale a vector
__global__
void scale(float a, float *x, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < n; i += stride) {
        x[i] *= a;
    }
}

void chatblas_sscal(int n, float a, float *x) {
    // set the size of the number of blocks and threads
    int blockSize = 256;
    int numOfBlocks = (n + blockSize - 1) / blockSize;

    scale<<<numOfBlocks, blockSize>>>(a, x, n);

    // Synchronize the CUDA device to make sure all threads have finished before continuing
    // This is generally only necessary for purposes of timing, error checking, etc.
    cudaDeviceSynchronize();
}
