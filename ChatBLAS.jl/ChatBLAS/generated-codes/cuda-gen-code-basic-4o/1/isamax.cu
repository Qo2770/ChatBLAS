#include "chatblas_cuda.h"

__global__ void find_max_abs_index(float *x, int *indices, int n) {
    extern __shared__ float shared_data[];
    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < n) {
        shared_data[tid] = fabsf(x[index]);
        indices[tid] = index;
    } else {
        shared_data[tid] = -1.0f; // use a negative value as default
        indices[tid] = -1;        // invalid index 
    }
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (shared_data[tid] < shared_data[tid + stride]) {
                shared_data[tid] = shared_data[tid + stride];
                indices[tid] = indices[tid + stride];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        x[blockIdx.x] = shared_data[0];
        indices[blockIdx.x] = indices[0];
    }
}

int chatblas_isamax(int n, float *x) {
    float *d_x;
    int *d_indices;
    int *h_indices = (int *)malloc(sizeof(int) * n);
    int num_blocks = (n + 1023) / 1024;

    cudaMalloc((void**)&d_x, n * sizeof(float));
    cudaMalloc((void**)&d_indices, n * sizeof(int));
    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);

    find_max_abs_index<<<num_blocks, 1024, 1024 * sizeof(float)>>>(d_x, d_indices, n);

    cudaMemcpy(h_indices, d_indices, num_blocks * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(x, d_x, num_blocks * sizeof(float), cudaMemcpyDeviceToHost);

    int max_idx = 0;
    float max_val = fabsf(x[0]);
    for (int i = 1; i < num_blocks; ++i) {
        if (fabsf(x[i]) > max_val) {
            max_val = fabsf(x[i]);
            max_idx = i;
        }
    }
    int idx = h_indices[max_idx];

    cudaFree(d_x);
    cudaFree(d_indices);
    free(h_indices);

    return idx;
}
