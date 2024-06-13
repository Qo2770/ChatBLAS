#include "chatblas_cuda.h"

__global__ void sdot_kernel(int n, float *x, float *y, float *res) {
    // Code for the kernel to compute dot product
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    __shared__ float shared_results[256];  // Assuming block size of 256 for simplicity

    float local_sum = 0.0f;
    for (int i = tid; i < n; i += stride) {
        local_sum += x[i] * y[i];
    }
    shared_results[threadIdx.x] = local_sum;

    __syncthreads();

    // Reduce the shared results within the block
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_results[threadIdx.x] += shared_results[threadIdx.x + s];
        }
        __syncthreads();
    }

    // Store the final result in the res pointer
    if (threadIdx.x == 0) {
        atomicAdd(res, shared_results[0]);
    }
}

float chatblas_sdot(int n, float *x, float *y) {
    float *d_x, *d_y, *d_res;
    float result;

    // Allocate GPU memory for vectors x, y, and res
    cudaMalloc((void **)&d_x, n * sizeof(float));
    cudaMalloc((void **)&d_y, n * sizeof(float));
    cudaMalloc((void **)&d_res, sizeof(float));

    // Copy data from CPU to GPU
    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_res, 0, sizeof(float));

    // Define the number of blocks and threads per block
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    // Call the kernel function
    sdot_kernel<<<blocksPerGrid, threadsPerBlock>>>(n, d_x, d_y, d_res);

    // Copy the result from GPU to CPU
    cudaMemcpy(&result, d_res, sizeof(float), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_res);

    return result;
}
