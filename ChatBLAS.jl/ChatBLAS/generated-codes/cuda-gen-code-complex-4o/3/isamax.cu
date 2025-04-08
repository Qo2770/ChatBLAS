#include "chatblas_cuda.h"

__global__ void isamax_kernel(int n, float *x, int *ind) {
    __shared__ float sdata[256];
    __shared__ int sind[256];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        sdata[tid] = fabs(x[i]);
        sind[tid] = i;
    } else {
        sdata[tid] = 0;
        sind[tid] = -1;
    }
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (sdata[tid] < sdata[tid + s] || (sdata[tid] == sdata[tid + s] && sind[tid] > sind[tid + s])) {
                sdata[tid] = sdata[tid + s];
                sind[tid] = sind[tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        x[blockIdx.x] = sdata[0];
        ind[blockIdx.x] = sind[0];
    }
}

int chatblas_isamax(int n, float *x) {
    float *d_x;
    int *d_ind;
    int *h_ind;
    int index = 0;

    cudaMalloc((void**)&d_x, n * sizeof(float));
    cudaMalloc((void**)&d_ind, n * sizeof(int));
    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    isamax_kernel<<<numBlocks, blockSize>>>(n, d_x, d_ind);

    h_ind = (int*)malloc(numBlocks * sizeof(int));
    cudaMemcpy(h_ind, d_ind, numBlocks * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 1; i < numBlocks; i++) {
        if (fabs(x[h_ind[index]]) < fabs(x[h_ind[i]])) {
            index = i;
        }
    }

    index = h_ind[index];
    free(h_ind);
    cudaFree(d_x);
    cudaFree(d_ind);

    return index;
}