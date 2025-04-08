#include "chatblas_cuda.h"

__global__ void isamax_kernel(int n, float *x, int *ind) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    sdata[tid] = (idx < n) ? fabsf(x[idx]) : -1.0;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && sdata[tid] < sdata[tid + s]) {
            sdata[tid] = sdata[tid + s];
            ind[tid] = ind[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        ind[blockIdx.x] = idx - blockDim.x + threadIdx.x;
    }
}

int chatblas_isamax(int n, float *x) {
    float *d_x;
    int *d_ind, *h_ind;
    int max_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    h_ind = (int *)malloc(max_blocks * sizeof(int));

    cudaMalloc((void **)&d_x, n * sizeof(float));
    cudaMalloc((void **)&d_ind, max_blocks * sizeof(int));

    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);

    isamax_kernel<<<max_blocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(n, d_x, d_ind);

    cudaMemcpy(h_ind, d_ind, max_blocks * sizeof(int), cudaMemcpyDeviceToHost);

    int max_ind = h_ind[0];
    for (int i = 1; i < max_blocks; i++) {
        if (fabs(x[h_ind[i]]) > fabs(x[max_ind])) {
            max_ind = h_ind[i];
        }
    }

    cudaFree(d_x);
    cudaFree(d_ind);
    free(h_ind);

    return max_ind;
}