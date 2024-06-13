#include "chatblas_cuda.h"

__global__ void copy_kernel(int n, float *x, float *y) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        y[idx] = x[idx];
}

void chatblas_scopy(int n, float *x, float *y) {
    // Allocate memory on the GPU
    float *d_x, *d_y;
    cudaMalloc((void **)&d_x, n * sizeof(float));
    cudaMalloc((void **)&d_y, n * sizeof(float));

    // Copy input data from host to device
    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);

    // Invoke kernel on the GPU
    int threads_per_block = 256;
    int num_blocks = (n + threads_per_block - 1) / threads_per_block;
    copy_kernel<<<num_blocks, threads_per_block>>>(n, d_x, d_y);

    // Copy result back from device to host
    cudaMemcpy(y, d_y, n * sizeof(float), cudaMemcpyDeviceToHost);

    // Free the GPU memory
    cudaFree(d_x);
    cudaFree(d_y);
}
