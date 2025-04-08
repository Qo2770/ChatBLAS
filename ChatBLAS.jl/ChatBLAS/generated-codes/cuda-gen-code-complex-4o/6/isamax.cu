#include "chatblas_cuda.h"

__global__ void isamax_kernel(int n, float *x, int *ind) {
    __shared__ float max_val[256];
    __shared__ int max_idx[256];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Initialize shared memory
    if (idx < n) {
        max_val[tid] = fabsf(x[idx]);
        max_idx[tid] = idx;
    } else {
        max_val[tid] = -1.0f;
        max_idx[tid] = -1;
    }
    
    __syncthreads();
    
    // Perform reduction on shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (max_val[tid] < max_val[tid + s]) {
                max_val[tid] = max_val[tid + s];
                max_idx[tid] = max_idx[tid + s];
            }
        }
        __syncthreads();
    }
    
    // Write the result of this block to global memory
    if (tid == 0) {
        ind[blockIdx.x] = max_idx[0];
    }
}

int chatblas_isamax(int n, float *x) {
    float *d_x;
    int *d_ind;
    int *h_ind;
    int max_index = 0;
    int num_blocks = (n + 255) / 256;
    
    cudaMalloc((void **)&d_x, n * sizeof(float));
    cudaMalloc((void **)&d_ind, num_blocks * sizeof(int));
    h_ind = (int *)malloc(num_blocks * sizeof(int));
    
    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
    
    isamax_kernel<<<num_blocks, 256>>>(n, d_x, d_ind);
    
    cudaMemcpy(h_ind, d_ind, num_blocks * sizeof(int), cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < num_blocks; i++) {
        if (fabsf(x[h_ind[i]]) > fabsf(x[max_index])) {
            max_index = h_ind[i];
        }
    }
    
    cudaFree(d_x);
    cudaFree(d_ind);
    free(h_ind);
    
    return max_index;
}