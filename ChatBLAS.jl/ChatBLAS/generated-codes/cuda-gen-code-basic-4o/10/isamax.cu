#include "chatblas_cuda.h"

__global__ void find_max_abs_index(float *x, int *max_index, int n) {
    extern __shared__ int shared_indices[];
    extern __shared__ float shared_values[];

    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + tid;

    if (index < n) {
        shared_indices[tid] = index;
        shared_values[tid] = fabsf(x[index]);
    } else {
        shared_indices[tid] = -1;
        shared_values[tid] = -FLT_MAX;
    }
    
    __syncthreads();

    // Parallel reduction to find the max absolute value and its index
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        __syncthreads();
        if (tid < stride) {
            if (shared_values[tid] < shared_values[tid + stride]) {
                shared_values[tid] = shared_values[tid + stride];
                shared_indices[tid] = shared_indices[tid + stride];
            }
        }
    }

    if (tid == 0) {
        max_index[blockIdx.x] = shared_indices[0];
    }
}

int chatblas_isamax(int n, float *x) {
    if (n <= 0) return -1;

    int threads_per_block = 256;
    int blocks = (n + threads_per_block - 1) / threads_per_block;

    float *d_x;
    int *d_max_index;
    int h_max_index[blocks];

    cudaMalloc((void**)&d_x, n * sizeof(float));
    cudaMalloc((void**)&d_max_index, blocks * sizeof(int));

    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);

    int shared_memory_size = 2 * threads_per_block * sizeof(float); // Shared memory for indices and values
    find_max_abs_index<<<blocks, threads_per_block, shared_memory_size>>>(d_x, d_max_index, n);

    cudaMemcpy(h_max_index, d_max_index, blocks * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_max_index);

    // Find the overall max index from the blocks
    int max_index = h_max_index[0];
    float max_value = fabsf(x[max_index]);
    for (int i = 1; i < blocks; i++) {
        int idx = h_max_index[i];
        float value = fabsf(x[idx]);
        if (value > max_value) {
            max_value = value;
            max_index = idx;
        }
    }

    return max_index;
}
