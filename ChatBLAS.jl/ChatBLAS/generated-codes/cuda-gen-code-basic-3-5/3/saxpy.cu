#include "chatblas_cuda.h"

__global__ void saxpy_kernel(int n, float a, float *x, float *y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = a * x[i] + y[i];
    }
}

void chatblas_saxpy(int n, float a, float *x, float *y) {
    float *d_x, *d_y;
    int num_bytes = n * sizeof(float);

    // Allocate memory on the GPU
    cudaMalloc((void **)&d_x, num_bytes);
    cudaMalloc((void **)&d_y, num_bytes);

    // Copy input vectors from host memory to GPU memory
    cudaMemcpy(d_x, x, num_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, num_bytes, cudaMemcpyHostToDevice);

    // Perform SAXPY computation in parallel on the GPU
    int block_size = 256;
    int num_blocks = (n + block_size - 1) / block_size;
    saxpy_kernel<<<num_blocks, block_size>>>(n, a, d_x, d_y);

    // Copy the result from GPU memory to host memory
    cudaMemcpy(y, d_y, num_bytes, cudaMemcpyDeviceToHost);

    // Free the allocated GPU memory
    cudaFree(d_x);
    cudaFree(d_y);
}
