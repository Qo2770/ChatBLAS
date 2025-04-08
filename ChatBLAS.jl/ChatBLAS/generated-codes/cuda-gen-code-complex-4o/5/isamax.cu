#include "chatblas_cuda.h"

__global__ void isamax_kernel(int n, float *x, int *ind) {
    extern __shared__ float sdata[];
    __shared__ int sind[256]; // assuming blockDim.x won't exceed 256

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    int gridSize = blockDim.x * gridDim.x;
    
    sdata[tid] = (idx < n) ? fabs(x[idx]) : -1.0f;
    sind[tid] = (idx < n) ? idx : -1;

    __syncthreads();

    for (int i = tid + blockDim.x; i < n; i += gridSize) {
        float x_value = fabs(x[i]);
        if (x_value > sdata[tid]) {
            sdata[tid] = x_value;
            sind[tid] = i;
        }
    }

    __syncthreads();

    // Reduction within block
    if (blockDim.x >= 512) { if (tid < 256 && sdata[tid] < sdata[tid + 256]) { sdata[tid] = sdata[tid + 256]; sind[tid] = sind[tid + 256]; } __syncthreads(); }
    if (blockDim.x >= 256) { if (tid < 128 && sdata[tid] < sdata[tid + 128]) { sdata[tid] = sdata[tid + 128]; sind[tid] = sind[tid + 128]; } __syncthreads(); }
    if (blockDim.x >= 128) { if (tid <  64 && sdata[tid] < sdata[tid +  64]) { sdata[tid] = sdata[tid +  64]; sind[tid] = sind[tid +  64]; } __syncthreads(); }

    if (tid < 32) {
        if (blockDim.x >=  64) { if (sdata[tid] < sdata[tid + 32]) { sdata[tid] = sdata[tid + 32]; sind[tid] = sind[tid + 32]; } }
        if (blockDim.x >=  32) { if (sdata[tid] < sdata[tid + 16]) { sdata[tid] = sdata[tid + 16]; sind[tid] = sind[tid + 16]; } }
        if (blockDim.x >=  16) { if (sdata[tid]  < sdata[tid + 8]) { sdata[tid]  = sdata[tid + 8]; sind[tid]  = sind[tid + 8]; } }
        if (blockDim.x >=   8) { if (sdata[tid]  < sdata[tid + 4]) { sdata[tid]  = sdata[tid + 4]; sind[tid]  = sind[tid + 4]; } }
        if (blockDim.x >=   4) { if (sdata[tid]  < sdata[tid + 2]) { sdata[tid]  = sdata[tid + 2]; sind[tid]  = sind[tid + 2]; } }
        if (blockDim.x >=   2) { if (sdata[tid]  < sdata[tid + 1]) { sdata[tid]  = sdata[tid + 1]; sind[tid]  = sind[tid + 1]; } }
    }

    if (tid == 0) {
        ind[blockIdx.x] = sind[0];
    }
}

int chatblas_isamax(int n, float *x) {
    int *d_ind, *h_ind;
    float *d_x;
    h_ind = (int*)malloc(sizeof(int) * 256);

    cudaMalloc(&d_x, n * sizeof(float));
    cudaMalloc(&d_ind, 256 * sizeof(int));

    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);

    int blocks = (n + 255) / 256;
    isamax_kernel<<<blocks, 256, 256 * sizeof(float)>>>(n, d_x, d_ind);

    cudaMemcpy(h_ind, d_ind, blocks * sizeof(int), cudaMemcpyDeviceToHost);

    int maxIndex = h_ind[0];
    for (int i = 1; i < blocks; ++i) {
        if (fabs(x[h_ind[i]]) > fabs(x[maxIndex])) {
            maxIndex = h_ind[i];
        }
    }

    cudaFree(d_x);
    cudaFree(d_ind);
    free(h_ind);

    return maxIndex;
}