#include "chatblas_cuda.h"

__global__ void isamax_kernel(int n, float *x, float *ind) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float max_value = -1.0f;
    int max_index = -1;
    
    if (i < n) {
        if (abs(x[i]) > max_value) {
            max_value = abs(x[i]);
            max_index = i;
        }
    }
    atomicMax(ind, max_index);
}

int chatblas_isamax(int n, float *x) {
    float *d_x, *d_ind;
    int max_index = -1;
    float *h_ind = &max_index;

    cudaMalloc((void**)&d_x, n * sizeof(float));
    cudaMalloc((void**)&d_ind, sizeof(float));

    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ind, h_ind, sizeof(float), cudaMemcpyHostToDevice);

    int blocks = ceil(n / 256.0);
    isamax_kernel <<< blocks, 256 >>> (n, d_x, d_ind);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_ind, d_ind, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_x);
    cudaFree(d_ind);

    return max_index;
}