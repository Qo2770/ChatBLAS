#include "chatblas_cuda.h"

__global__ void sdot_kernel(int n, float *x, float *y, float *res) {
    // Shared memory to hold partial dot products
    __shared__ float sdata[256];
    
    // Calculate global thread index
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int gridSize = blockDim.x * gridDim.x;
    
    // Initialize partial dot product to 0
    float temp = 0.0f;
    
    // Perform dot product in parallel
    while (tid < n) {
        temp += x[tid] * y[tid];
        tid += gridSize;
    }
    
    // Store the partial dot product in shared memory
    sdata[threadIdx.x] = temp;
    
    // Sync threads within the block
    __syncthreads();
    
    // Perform reduction by adding partial dot products
    int i = blockDim.x / 2;
    while (i != 0) {
        if (threadIdx.x < i) {
            sdata[threadIdx.x] += sdata[threadIdx.x + i];
        }
        __syncthreads();
        i /= 2;
    }
    
    // The first thread writes the final result to global memory
    if (threadIdx.x == 0) {
        *res = sdata[0];
    }
}

float chatblas_sdot(int n, float *x, float *y) {
    // Allocate memory for vectors on GPU
    float *d_x, *d_y;
    cudaMalloc((void**)&d_x, n * sizeof(float));
    cudaMalloc((void**)&d_y, n * sizeof(float));
    
    // Copy vectors from CPU to GPU
    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, n * sizeof(float), cudaMemcpyHostToDevice);
    
    // Allocate memory for result on GPU
    float *d_res;
    cudaMalloc((void**)&d_res, sizeof(float));
    
    // Number of threads per block and number of blocks
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    
    // Launch the kernel
    sdot_kernel<<<numBlocks, blockSize>>>(n, d_x, d_y, d_res);
    
    // Copy the result from GPU to CPU
    float res;
    cudaMemcpy(&res, d_res, sizeof(float), cudaMemcpyDeviceToHost);
    
    // Free GPU memory
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_res);
    
    // Return the dot product result
    return res;
}
