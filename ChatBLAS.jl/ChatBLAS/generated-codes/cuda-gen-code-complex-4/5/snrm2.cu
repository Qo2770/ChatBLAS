#include "chatblas_cuda.h"

__global__ void snrm2_kernel(int n, float *x, float *res) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    float temp = 0.0;
    if (idx < n) {
        temp = x[idx];
    }
    __syncthreads();

    atomicAdd(res, temp * temp);
    __syncthreads();
}

float chatblas_snrm2(int n, float *x) {
    float *x_d, *res, result;

    cudaMalloc(&x_d, sizeof(float) * n);
    cudaMalloc(&res, sizeof(float));

    cudaMemcpy(x_d, x, sizeof(float) * n, cudaMemcpyHostToDevice);

    float initial_value = 0.0;
    cudaMemcpy(res, &initial_value, sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    snrm2_kernel <<<blocksPerGrid, threadsPerBlock>>> (n, x_d, res);

    cudaMemcpy(&result, res, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(x_d);
    cudaFree(res);

    return sqrt(result);
}