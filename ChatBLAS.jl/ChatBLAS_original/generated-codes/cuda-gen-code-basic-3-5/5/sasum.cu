#include "chatblas_cuda.h"

__global__ void sasum_kernel(int n, float *x, float *result) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n) {
        // Compute absolute value and add to the result
        atomicAdd(result, fabsf(x[tid]));
    }
}

float chatblas_sasum(int n, float *x) {
    // Allocate memory on the GPU for x and the result
    float *d_x, h_result, *d_result;
    cudaMalloc((void **)&d_x, n * sizeof(float));
    cudaMalloc((void **)&d_result, sizeof(float));

    // Copy input vector x from host to the GPU memory
    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);

    // Set the initial value of the result to 0 on the GPU
    cudaMemcpy(d_result, &h_result, sizeof(float), cudaMemcpyHostToDevice);

    // Number of threads per block
    int threadsPerBlock = 256;

    // Number of blocks needed
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the kernel
    sasum_kernel<<<blocksPerGrid, threadsPerBlock>>>(n, d_x, d_result);

    // Copy the result from GPU memory to host
    cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    
    // Free the GPU memory
    cudaFree(d_x);
    cudaFree(d_result);

    return h_result;
}
