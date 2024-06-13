#include "chatblas_cuda.h"

__global__ void saxpy_kernel(int n, float a, float *x, float *y) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) y[i] = a*x[i] + y[i];
}

void chatblas_saxpy(int n, float a, float *x, float *y) {
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    saxpy_kernel<<<numBlocks, blockSize>>>(n, a, x, y);

    // Check for any errors launching the kernel
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error launching saxpy_kernel: %s\n", cudaGetErrorString(err));
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)  {
        printf("cudaDeviceSynchronize error: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}
