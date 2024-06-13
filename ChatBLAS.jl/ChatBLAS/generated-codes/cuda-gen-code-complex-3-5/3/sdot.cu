#include "chatblas_cuda.h"

__global__ void sdot_kernel(int n, float *x, float *y, float *res) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    float sum = 0.0f;
    for (int i = index; i < n; i += stride) {
        sum += x[i] * y[i];
    }

    atomicAdd(res, sum);
}

float chatblas_sdot(int n, float *x, float *y) {
    float *d_x, *d_y, *d_res;

    // Allocate memory on the GPU
    cudaMalloc((void **)&d_x, n * sizeof(float));
    cudaMalloc((void **)&d_y, n * sizeof(float));
    cudaMalloc((void **)&d_res, sizeof(float));

    // Copy input vectors from host memory to GPU buffers
    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, n * sizeof(float), cudaMemcpyHostToDevice);

    // Set initial value of result to 0 on GPU
    cudaMemset(d_res, 0, sizeof(float));

    // Define the number of threads per block and the number of blocks
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the kernel on GPU
    sdot_kernel<<<blocksPerGrid, threadsPerBlock>>>(n, d_x, d_y, d_res);

    // Copy the result from GPU buffer to host memory
    float res;
    cudaMemcpy(&res, d_res, sizeof(float), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_res);

    return res;
}
