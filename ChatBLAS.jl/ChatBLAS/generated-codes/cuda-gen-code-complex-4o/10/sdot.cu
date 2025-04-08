#include "chatblas_cuda.h"

__global__ void sdot_kernel(int n, float *x, float *y, float *res) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int tidx = threadIdx.x;

    // Load elements into shared memory
    float tmp = 0.0f;
    if (tid < n) {
        tmp = x[tid] * y[tid];
    }
    sdata[tidx] = tmp;
    __syncthreads();

    // Reduce sum in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tidx < s) {
            sdata[tidx] += sdata[tidx + s];
        }
        __syncthreads();
    }

    // Write result of block to global memory
    if (tidx == 0) {
        atomicAdd(res, sdata[0]);
    }
}

float chatblas_sdot(int n, float *x, float *y) {
    float *d_x, *d_y, *d_res;
    float result = 0.0f;
    float *h_res = &result;

    cudaMalloc((void **)&d_x, n * sizeof(float));
    cudaMalloc((void **)&d_y, n * sizeof(float));
    cudaMalloc((void **)&d_res, sizeof(float));

    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_res, h_res, sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256; // Can be adjusted depending on the GPU
    int numBlocks = (n + blockSize - 1) / blockSize;
    size_t sharedMemSize = blockSize * sizeof(float);

    sdot_kernel<<<numBlocks, blockSize, sharedMemSize>>>(n, d_x, d_y, d_res);
    cudaMemcpy(h_res, d_res, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_res);

    return result;
}