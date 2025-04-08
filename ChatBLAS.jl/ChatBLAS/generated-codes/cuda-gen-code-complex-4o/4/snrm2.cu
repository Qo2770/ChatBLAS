#include "chatblas_cuda.h"

__global__ void snrm2_kernel(int n, float *x, float *res) {
    __shared__ float partialSum[256];
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    float sum = 0.0f;

    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        sum += x[i] * x[i];
    }

    partialSum[threadIdx.x] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            partialSum[threadIdx.x] += partialSum[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(res, partialSum[0]);
    }
}

float chatblas_snrm2(int n, float *x) {
    float *d_x, *d_res;
    float h_res = 0.0f;
    int size = n * sizeof(float);

    cudaMalloc((void**)&d_x, size);
    cudaMalloc((void**)&d_res, sizeof(float));

    cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_res, &h_res, sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    snrm2_kernel<<<blocksPerGrid, threadsPerBlock>>>(n, d_x, d_res);
    cudaDeviceSynchronize();

    cudaMemcpy(&h_res, d_res, sizeof(float), cudaMemcpyDeviceToHost);

    h_res = sqrt(h_res);

    cudaFree(d_x);
    cudaFree(d_res);

    return h_res;
}