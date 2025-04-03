#include "chatblas_cuda.h"

__global__ void sasum_kernel(int n, float *x, float *sum) {
    extern __shared__ float sdata[]; 
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int gridSize = blockDim.x * gridDim.x;
    
    float partialSum = 0.0f;
    while (tid < n) {
        partialSum += fabsf(x[tid]);
        tid += gridSize;
    }
    
    sdata[threadIdx.x] = partialSum;
    __syncthreads();

    int i = blockDim.x/2;
    while (i != 0) {
        if (threadIdx.x < i) {
            sdata[threadIdx.x] += sdata[threadIdx.x + i];
        }
        __syncthreads();
        i /= 2;
    }

    if (threadIdx.x == 0) {
        atomicAdd(sum, sdata[0]);
    }
}

float chatblas_sasum(int n, float *x) {
    float *d_x, sum;
    cudaMalloc((void**)&d_x, n * sizeof(float));
    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block_size(256);
    dim3 num_blocks((n - 1) / block_size.x + 1);

    sasum_kernel<<<num_blocks, block_size, block_size.x * sizeof(float)>>>(n, d_x, &sum);
    cudaMemcpy(&sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_x);

    return sum;
}
