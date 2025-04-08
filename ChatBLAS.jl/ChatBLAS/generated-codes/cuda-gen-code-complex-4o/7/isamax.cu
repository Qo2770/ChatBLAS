#include "chatblas_cuda.h"

__global__ void isamax_kernel(int n, float *x, int *ind) {
    extern __shared__ float shared_x[];
    extern __shared__ int shared_ind[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    shared_x[tid] = (idx < n) ? fabsf(x[idx]) : -1.0f;
    shared_ind[tid] = idx;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            if (shared_x[tid] < shared_x[tid + stride]) {
                shared_x[tid] = shared_x[tid + stride];
                shared_ind[tid] = shared_ind[tid + stride];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        ind[blockIdx.x] = shared_ind[0];
    }
}

int chatblas_isamax(int n, float *x) {
    float *d_x;
    int *d_ind, *h_ind;
    int num_blocks = (n + 255) / 256;
    int block_size = 256;

    cudaMalloc((void**)&d_x, n * sizeof(float));
    cudaMalloc((void**)&d_ind, num_blocks * sizeof(int));
    h_ind = (int*)malloc(num_blocks * sizeof(int));

    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);

    isamax_kernel<<<num_blocks, block_size, block_size * sizeof(float)>>>(n, d_x, d_ind);

    cudaMemcpy(h_ind, d_ind, num_blocks * sizeof(int), cudaMemcpyDeviceToHost);

    int max_idx = h_ind[0];
    for (int i = 1; i < num_blocks; ++i) {
        if (fabs(x[h_ind[i]]) > fabs(x[max_idx])) {
            max_idx = h_ind[i];
        }
    }

    cudaFree(d_x);
    cudaFree(d_ind);
    free(h_ind);

    return max_idx;
}