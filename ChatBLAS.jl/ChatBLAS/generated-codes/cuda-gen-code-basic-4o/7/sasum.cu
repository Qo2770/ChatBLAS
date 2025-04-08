#include "chatblas_cuda.h"

__global__ void sasum_kernel(int n, float *x, float *result) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + tid;
    float sum = 0.0f;

    // Accumulate absolute values in shared memory
    if (index < n) {
        sum = fabsf(x[index]);
    }
    sdata[tid] = sum;
    __syncthreads();

    // Reduce within block
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write block's result to global memory
    if (tid == 0) {
        atomicAdd(result, sdata[0]);
    }
}

float chatblas_sasum(int n, float *x) {
    float *d_x, *d_result;
    float h_result = 0.0f;

    // Allocate memory on the device
    cudaMalloc((void**)&d_x, n * sizeof(float));
    cudaMalloc((void**)&d_result, sizeof(float));
    
    // Copy data and initialize result on the device
    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_result, &h_result, sizeof(float), cudaMemcpyHostToDevice);

    // Kernel launch parameters
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    // Launch kernel
    sasum_kernel<<<gridSize, blockSize, blockSize * sizeof(float)>>>(n, d_x, d_result);

    // Copy the result back to the host
    cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_x);
    cudaFree(d_result);

    return h_result;
}
