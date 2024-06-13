#include "chatblas_cuda.h"

__global__ void sdsdot_kernel(int n, float b, float *x, float *y, float *res) {
    // Compute the dot product of vectors x and y plus scalar b
    double sum = 0.0;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
  
    for (int i = tid; i < n; i += stride) {
        sum += (double)x[i] * y[i];
    }
  
    // Accumulate the result and store in res
    atomicAdd(res, b + sum);
}

float chatblas_sdsdot(int n, float b, float *x, float *y) {
    // Allocate memory for GPU vectors x, y, and res
    float *d_x, *d_y, *d_res;
    cudaMalloc((void **)&d_x, n * sizeof(float));
    cudaMalloc((void **)&d_y, n * sizeof(float));
    cudaMalloc((void **)&d_res, sizeof(float));

    // Copy vectors x and y from CPU to GPU
    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, n * sizeof(float), cudaMemcpyHostToDevice);

    // Set number of threads per block and number of blocks
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    // Call the kernel function
    sdsdot_kernel<<<blocksPerGrid, threadsPerBlock>>>(n, b, d_x, d_y, d_res);

    // Copy result from GPU to CPU
    float result;
    cudaMemcpy(&result, d_res, sizeof(float), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_res);

    // Return the computed result
    return result;
}
