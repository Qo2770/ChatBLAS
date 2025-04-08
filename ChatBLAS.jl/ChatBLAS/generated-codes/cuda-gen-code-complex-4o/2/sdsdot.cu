#include "chatblas_cuda.h"

__global__ void sdsdot_kernel(int n, float b, float *x, float *y, float *res) {
    extern __shared__ double sdata[];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int blockSize = blockDim.x;
    int tx = threadIdx.x;
    
    double temp = 0.0;
    while (tid < n) {
        temp += (double)x[tid] * (double)y[tid];
        tid += blockDim.x * gridDim.x;
    }

    sdata[tx] = temp;
    __syncthreads();

    for (unsigned int s = blockSize / 2; s > 0; s >>= 1) {
        if (tx < s) {
            sdata[tx] += sdata[tx + s];
        }
        __syncthreads();
    }

    if (tx == 0) {
        atomicAdd(res, sdata[0]);
    }
}

float chatblas_sdsdot(int n, float b, float *x, float *y) {
    float *d_x, *d_y, *d_res;
    float host_res = 0.0f;
    cudaMalloc((void**)&d_x, n * sizeof(float));
    cudaMalloc((void**)&d_y, n * sizeof(float));
    cudaMalloc((void**)&d_res, sizeof(float));
    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_res, &host_res, sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    sdsdot_kernel<<<gridSize, blockSize, blockSize * sizeof(double)>>>(n, b, d_x, d_y, d_res);
    
    cudaMemcpy(&host_res, d_res, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_res);
    
    return host_res + b;
}