#include "chatblas_hip.h"

__global__ void sasum_kernel(int n, float *x, float *sum) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        atomicAdd(sum, fabsf(x[idx]));
    }
}

float chatblas_sasum(int n, float *x) {
    float *d_x, h_sum, *d_sum;

    // Allocate memory on the GPU
    hipMalloc(&d_x, n * sizeof(float));
    hipMalloc(&d_sum, sizeof(float));

    // Copy data from host to device
    hipMemcpy(d_x, x, n * sizeof(float), hipMemcpyHostToDevice);

    // Initialize sum to 0 on GPU
    hipMemset(d_sum, 0, sizeof(float);

    // Call kernel
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    sasum_kernel<<<numBlocks, blockSize>>>(n, d_x, d_sum);

    // Copy result back to host
    hipMemcpy(&h_sum, d_sum, sizeof(float), hipMemcpyDeviceToHost);

    // Free memory on GPU
    hipFree(d_x);
    hipFree(d_sum);

    return h_sum;
}
