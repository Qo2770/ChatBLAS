#include "chatblas_cuda.h"
__global__ void sasum_kernel(int n, float *x, float *sum) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = gridDim.x * blockDim.x;
    sdata[tid] = 0.0;

    while (index < n) {
        sdata[tid] += fabsf(x[index]);
        index += stride;
    }

    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(sum, sdata[0]);
    }
}

float chatblas_sasum(int n, float *x) {
    float *d_x, *d_sum;
    float h_sum = 0.0;

    cudaMalloc((void**)&d_x, n * sizeof(float));
    cudaMalloc((void**)&d_sum, sizeof(float));

    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sum, &h_sum, sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    sasum_kernel<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(n, d_x, d_sum);

    cudaMemcpy(&h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_sum);

    return h_sum;
}