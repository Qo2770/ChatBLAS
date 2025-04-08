#include "chatblas_cuda.h"

__global__ void isamax_kernel(int n, float *x, int *ind) {
    extern __shared__ float sdata[];
    __shared__ int sind[256];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        sdata[tid] = fabsf(x[i]);
        sind[tid] = i;
    } else {
        sdata[tid] = -1.0f;
    }

    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && i + s < n) {
            if (sdata[tid] < sdata[tid + s]) {
                sdata[tid] = sdata[tid + s];
                sind[tid] = sind[tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        if (sdata[0] > sdata[blockIdx.x * blockDim.x]) {
            if (atomicCAS(ind, *ind, sind[0]) == *ind) {
                atomicExch(ind, sind[0]);
            }
        }
    }
}

int chatblas_isamax(int n, float *x) {
    float *d_x;
    int *d_ind;
    int h_ind;
    int blocks = (n + 255) / 256;
    int threads = 256;

    cudaMalloc((void**)&d_x, n * sizeof(float));
    cudaMalloc((void**)&d_ind, sizeof(int));

    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_ind, 0, sizeof(int));

    isamax_kernel<<<blocks, threads, threads * sizeof(float)>>>(n, d_x, d_ind);
    cudaMemcpy(&h_ind, d_ind, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_ind);

    return h_ind;
}