#include "chatblas_cuda.h"

__global__ void chatblas_scopy_device(int n, float *x, float *y) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    if (index < n) {
        y[index] = x[index];
    }
}

extern "C" {
    void chatblas_scopy(int n, float *x, float *y) {
        // compute number of blocks needed
        int blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

        // allocate device memory
        float *d_x, *d_y;
        cudaMalloc((void**) &d_x, sizeof(float) * n);
        cudaMalloc((void**) &d_y, sizeof(float) * n);

        // copy input to device
        cudaMemcpy(d_x, x, sizeof(float) * n, cudaMemcpyHostToDevice);

        // Execute copy kernel
        chatblas_scopy_device<<<blocks, THREADS_PER_BLOCK>>>(n, d_x, d_y);
        
        // copy result back to host
        cudaMemcpy(y, d_y, sizeof(float) * n, cudaMemcpyDeviceToHost);

        // release device memory
        cudaFree(d_x);
        cudaFree(d_y);
    }
}
