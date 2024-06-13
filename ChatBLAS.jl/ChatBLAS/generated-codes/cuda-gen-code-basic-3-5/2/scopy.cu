#include "chatblas_cuda.h"

__global__ void copy_kernel(int n, float *x, float *y) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n) {
        y[tid] = x[tid];
    }
}

void chatblas_scopy(int n, float *x, float *y) {
    // Allocate memory on device
    float *d_x, *d_y;
    cudaMalloc((void **)&d_x, n * sizeof(float));
    cudaMalloc((void **)&d_y, n * sizeof(float));

    // Copy input vectors from host to device
    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    int block_size = 256;
    int num_blocks = (n + block_size - 1) / block_size;
    copy_kernel<<<num_blocks, block_size>>>(n, d_x, d_y);

    // Copy result back to host
    cudaMemcpy(y, d_y, n * sizeof(float), cudaMemcpyDeviceToHost);

    // Free memory on device
    cudaFree(d_x);
    cudaFree(d_y);
}
