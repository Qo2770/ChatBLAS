#include "chatblas_cuda.h"

__global__ void compute_partial_norm(int n, float *x, float *partial_norm) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < n) {
        atomicAdd(partial_norm, x[idx] * x[idx]);
    }
}

float chatblas_snrm2(int n, float *x) {
    if (n <= 0) return 0.0f;

    // Allocate device memory
    float *d_x, *d_partial_norm;
    float h_partial_norm = 0.0f;

    cudaMalloc((void**)&d_x, n * sizeof(float));
    cudaMalloc((void**)&d_partial_norm, sizeof(float));

    // Copy data to device
    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_partial_norm, &h_partial_norm, sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel to compute partial norms
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    compute_partial_norm<<<numBlocks, blockSize>>>(n, d_x, d_partial_norm);

    // Copy result back to host
    cudaMemcpy(&h_partial_norm, d_partial_norm, sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_x);
    cudaFree(d_partial_norm);

    // Compute square root of accumulated norm
    return sqrt(h_partial_norm);
}
