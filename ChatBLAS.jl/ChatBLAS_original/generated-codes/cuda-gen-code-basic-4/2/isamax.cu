__global__ void chatblas_isamax_kernel(int n, float *x, int *imax) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < n) {
        float this_val = fabsf( x[idx] );
        float max_val = this_val;

        for (int i = idx + blockDim.x * gridDim.x; i < n; i += blockDim.x * gridDim.x) {
            this_val = fabsf( x[i] );
            if (this_val > max_val) {
                max_val = this_val;
                imax[0] = i;
            }
        }
    }
}
#include "chatblas_cuda.h"
#include <float.h>
#include <cmath>

#define THREADS_PER_BLOCK 512

int chatblas_isamax(int n, float *x) {
    int *max_index_device, max_index_host = 0;

    cudaMalloc(&max_index_device, sizeof(int));
    cudaMemcpy(max_index_device, &max_index_host, sizeof(int), cudaMemcpyHostToDevice);

    chatblas_isamax_kernel<<<(n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(n, x, max_index_device);

    cudaMemcpy(&max_index_host, max_index_device, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(max_index_device);

    return max_index_host;
}
