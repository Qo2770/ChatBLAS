#include "chatblas_cuda.h"

__global__ void isamax_kernel(int n, float *x, int *ind) {
    __shared__ float max_val_shared[256];
    __shared__ int max_idx_shared[256];
    
    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + tid;

    float max_val = -1.0f;
    int max_idx = -1;

    if (index < n) {
        float val = fabsf(x[index]);
        if (val > max_val) {
            max_val = val;
            max_idx = index;
        }
    }

    max_val_shared[tid] = max_val;
    max_idx_shared[tid] = max_idx;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (max_val_shared[tid + stride] > max_val_shared[tid]) {
                max_val_shared[tid] = max_val_shared[tid + stride];
                max_idx_shared[tid] = max_idx_shared[tid + stride];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        if (max_val_shared[0] > fabsf(x[*ind])) {
            *ind = max_idx_shared[0];
        }
    }
}

int chatblas_isamax(int n, float *x) {
    float *d_x;
    int *d_ind;
    int h_ind = 0;

    cudaMalloc((void**)&d_x, n * sizeof(float));
    cudaMalloc((void**)&d_ind, sizeof(int));

    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ind, &h_ind, sizeof(int), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    isamax_kernel<<<numBlocks, blockSize>>>(n, d_x, d_ind);

    cudaMemcpy(&h_ind, d_ind, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_ind);

    return h_ind;
}