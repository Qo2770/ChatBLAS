#include "chatblas_cuda.h"

__global__ void isamax_kernel(int n, float *x, int *ind) {
    __shared__ float abs_values[256];
    __shared__ int indices[256];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    
    if (i < n) {
        abs_values[tid] = fabsf(x[i]);
        indices[tid] = i;
    } else {
        abs_values[tid] = 0.0;
        indices[tid] = -1;
    }
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (abs_values[tid] < abs_values[tid + s]) {
                abs_values[tid] = abs_values[tid + s];
                indices[tid] = indices[tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicMax(ind, indices[0]);
    }
}

int chatblas_isamax(int n, float *x) {
    float *d_x;
    int *d_ind, h_ind;
    int size_x = n * sizeof(float);
    int size_ind = sizeof(int);

    cudaMalloc(&d_x, size_x);
    cudaMalloc(&d_ind, size_ind);
    cudaMemcpy(d_x, x, size_x, cudaMemcpyHostToDevice);
    cudaMemset(d_ind, 0, size_ind);

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    isamax_kernel<<<numBlocks, blockSize>>>(n, d_x, d_ind);

    cudaMemcpy(&h_ind, d_ind, size_ind, cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_ind);

    return h_ind;
}