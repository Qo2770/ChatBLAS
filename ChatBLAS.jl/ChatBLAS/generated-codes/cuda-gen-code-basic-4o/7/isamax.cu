#include "chatblas_cuda.h"

__global__ void find_max_abs_index(float *x, int n, int *index, float *max_val) {
    extern __shared__ float shared[];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    float *shared_vals = shared;
    int *shared_indices = (int*)&shared_vals[blockDim.x];

    if (gid < n) {
        shared_vals[tid] = fabsf(x[gid]);
        shared_indices[tid] = gid;
    } else {
        shared_vals[tid] = -1.0f; // Initialize to a negative value outside of the possible range
        shared_indices[tid] = -1;
    }
    __syncthreads();

    // Perform reduction to find the max absolute value and its index
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (shared_vals[tid] < shared_vals[tid + stride]) {
                shared_vals[tid] = shared_vals[tid + stride];
                shared_indices[tid] = shared_indices[tid + stride];
            }
        }
        __syncthreads();
    }

    // Write the result of this block
    if (tid == 0) {
        max_val[blockIdx.x] = shared_vals[0];
        index[blockIdx.x] = shared_indices[0];
    }
}

int chatblas_isamax(int n, float *x) {
    if (n <= 0) return -1;

    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;

    float *d_x;
    int *d_index, *h_index;
    float *d_max_val, *h_max_val;

    cudaMalloc((void**)&d_x, n * sizeof(float));
    cudaMalloc((void**)&d_index, grid_size * sizeof(int));
    cudaMalloc((void**)&d_max_val, grid_size * sizeof(float));

    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);

    h_index = (int*)malloc(grid_size * sizeof(int));
    h_max_val = (float*)malloc(grid_size * sizeof(float));

    find_max_abs_index<<<grid_size, block_size, block_size * (sizeof(float) + sizeof(int))>>>(d_x, n, d_index, d_max_val);

    cudaMemcpy(h_index, d_index, grid_size * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_max_val, d_max_val, grid_size * sizeof(float), cudaMemcpyDeviceToHost);

    int max_index = h_index[0];
    float max_value = h_max_val[0];

    for (int i = 1; i < grid_size; i++) {
        if (h_max_val[i] > max_value) {
            max_value = h_max_val[i];
            max_index = h_index[i];
        }
    }

    cudaFree(d_x);
    cudaFree(d_index);
    cudaFree(d_max_val);
    free(h_index);
    free(h_max_val);

    return max_index;
}
