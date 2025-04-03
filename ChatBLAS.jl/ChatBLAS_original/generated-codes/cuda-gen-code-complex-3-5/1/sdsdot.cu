#include "chatblas_cuda.h"

__global__ void sdsdot_kernel(int n, float b, float *x, float *y, double *res) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    double acc = 0.0;
    while (idx < n) {
        acc += double(x[idx]) * double(y[idx]);
        idx += blockDim.x * gridDim.x;
    }

    atomicAdd(res, acc + double(b));
}

float chatblas_sdsdot(int n, float b, float *x, float *y) {
    // Allocate GPU memory for the vectors
    float *d_x, *d_y;
    cudaMalloc((void **)&d_x, n * sizeof(float));
    cudaMalloc((void **)&d_y, n * sizeof(float));

    // Copy the vectors from CPU to GPU memory
    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, n * sizeof(float), cudaMemcpyHostToDevice);

    // Define block size and number of blocks
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    // Allocate GPU memory for the result
    double *d_res;
    cudaMalloc((void **)&d_res, sizeof(double));
    cudaMemset(d_res, 0, sizeof(double));

    // Launch the kernel
    sdsdot_kernel<<<numBlocks, blockSize>>>(n, b, d_x, d_y, d_res);

    // Copy the result from GPU to CPU memory
    double res;
    cudaMemcpy(&res, d_res, sizeof(double), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_res);

    return float(res);
}