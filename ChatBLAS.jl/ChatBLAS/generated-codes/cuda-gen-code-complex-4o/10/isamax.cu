#include "chatblas_cuda.h"

__global__ void isamax_kernel(int n, float *x, int *ind) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < n) sdata[tid] = fabsf(x[index]);
    else sdata[tid] = 0.0f;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (sdata[tid] < sdata[tid + s]) {
                sdata[tid] = sdata[tid + s];
                ind[tid] = index + s;
            }
        }
        __syncthreads();
    }

    if (tid == 0) ind[blockIdx.x] = ind[0];
}

int chatblas_isamax(int n, float *x) {
    float *d_x;
    int *d_ind, *h_ind;
    int blocks = 256;
    int threads = 256;
    size_t size_ind = blocks * sizeof(int);

    cudaMalloc((void **)&d_x, n * sizeof(float));
    cudaMalloc((void **)&d_ind, size_ind);
    h_ind = (int *)malloc(size_ind);

    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);

    isamax_kernel<<<blocks, threads, threads * sizeof(float)>>>(n, d_x, d_ind);

    cudaMemcpy(h_ind, d_ind, size_ind, cudaMemcpyDeviceToHost);

    int max_index = 0;
    float max_val = 0.0f;
    for (int i = 0; i < blocks; i++) {
        if (fabsf(x[h_ind[i]]) > max_val) {
            max_val = fabsf(x[h_ind[i]]);
            max_index = h_ind[i];
        }
    }

    cudaFree(d_x);
    cudaFree(d_ind);
    free(h_ind);

    return max_index;
}