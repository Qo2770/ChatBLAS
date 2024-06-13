#include "chatblas_cuda.h"

__global__ void snrm2_kernel(int n, float *x, float *res) {
    // Determine the index of the current thread
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Declare a shared memory array for partial sums
    __shared__ float partial_sums[256];

    // Initialize partial sum to 0 for each thread
    partial_sums[threadIdx.x] = 0.0f;

    // Compute partial sum for each thread
    if (idx < n) {
        float value = x[idx];
        partial_sums[threadIdx.x] = value * value;
    }

    // Perform parallel reduction to get the final sum
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        __syncthreads();
        if (threadIdx.x < stride) {
            partial_sums[threadIdx.x] += partial_sums[threadIdx.x + stride];
        }
    }

    // Store the computed Euclidean norm in the pointer res
    if (threadIdx.x == 0) {
        res[blockIdx.x] = partial_sums[0];
    }
}

float chatblas_snrm2(int n, float *x) {
    // Number of elements per block
    int block_size = 256;

    // Compute the number of blocks required
    int num_blocks = (n + block_size - 1) / block_size;

    // Allocate CPU memory for the result
    float *res_cpu = (float *)malloc(num_blocks * sizeof(float));
    
    // Allocate GPU memory for the vectors
    float *x_gpu, *res_gpu;
    cudaMalloc((void **)&x_gpu, n * sizeof(float));
    cudaMalloc((void **)&res_gpu, num_blocks * sizeof(float));

    // Transfer input vector from CPU to GPU
    cudaMemcpy(x_gpu, x, n * sizeof(float), cudaMemcpyHostToDevice);

    // Launch the kernel
    snrm2_kernel<<<num_blocks, block_size>>>(n, x_gpu, res_gpu);

    // Transfer result vector from GPU to CPU
    cudaMemcpy(res_cpu, res_gpu, num_blocks * sizeof(float), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(x_gpu);
    cudaFree(res_gpu);

    // Compute the final Euclidean norm
    float result = 0.0f;
    for (int i = 0; i < num_blocks; i++) {
        result += res_cpu[i];
    }
    result = sqrt(result);

    // Free CPU memory
    free(res_cpu);
    
    return result;
}