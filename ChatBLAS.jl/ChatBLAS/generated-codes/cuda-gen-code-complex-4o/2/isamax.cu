#include "chatblas_cuda.h"

__global__ void isamax_kernel(int n, float *x, int *ind) {
    extern __shared__ float sdata[];
    __shared__ int sindex[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (i < n) ? fabsf(x[i]) : 0.0f;
    sindex[tid] = i;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (sdata[tid] < sdata[tid + s]) {
                sdata[tid] = sdata[tid + s];
                sindex[tid] = sindex[tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        if (blockIdx.x == 0 || sdata[0] > x[ind[0]]) {
            ind[0] = sindex[0];
            x[ind[0]] = sdata[0];
        }
    }
}

int chatblas_isamax(int n, float *x) {
    float *d_x;
    int *d_ind;
    int ind_host;

    cudaMalloc((void**)&d_x, n * sizeof(float));
    cudaMalloc((void**)&d_ind, sizeof(int));

    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ind, &ind_host, sizeof(int), cudaMemcpyHostToDevice);

    int numBlocks = (n + 255) / 256;
    int blockSize = 256;
    isamax_kernel<<<numBlocks, blockSize, blockSize * sizeof(float)>>>(n, d_x, d_ind);

    cudaMemcpy(&ind_host, d_ind, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_ind);

    return ind_host;
}