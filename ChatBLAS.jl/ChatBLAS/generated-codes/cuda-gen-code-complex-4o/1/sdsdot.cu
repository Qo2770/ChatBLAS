#include "chatblas_cuda.h"

__global__ void sdsdot_kernel( int n, float b, float *x, float *y, float *res ) {
    extern __shared__ double temp[];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int thread_lane = threadIdx.x;
    
    double local_sum = 0.0;
    
    while (tid < n) {
        local_sum += static_cast<double>(x[tid]) * static_cast<double>(y[tid]);
        tid += blockDim.x * gridDim.x;
    }
    
    temp[thread_lane] = local_sum;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (thread_lane < stride) {
            temp[thread_lane] += temp[thread_lane + stride];
        }
        __syncthreads();
    }
    
    if (thread_lane == 0) {
        atomicAdd(res, temp[0]);
    }
}

float chatblas_sdsdot( int n, float b, float *x, float *y) {
    float *d_x, *d_y, *d_res;
    float h_res = 0.0f;
    cudaMalloc((void **)&d_x, n * sizeof(float));
    cudaMalloc((void **)&d_y, n * sizeof(float));
    cudaMalloc((void **)&d_res, sizeof(float));
    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_res, &h_res, sizeof(float), cudaMemcpyHostToDevice);
    
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    sdsdot_kernel<<<numBlocks, blockSize, blockSize * sizeof(double)>>>(n, b, d_x, d_y, d_res);
    
    cudaMemcpy(&h_res, d_res, sizeof(float), cudaMemcpyDeviceToHost);
    
    h_res += b;
    
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_res);
    
    return h_res;
}