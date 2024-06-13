#include "chatblas_hip.h"

__global__ void isamax_kernel(int n, float *x, int *ind) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < n) {
        atomicMax(&ind[0], fabsf(x[tid]) > fabsf(x[ind[0]]) ? tid : ind[0]);
    }
}

int chatblas_isamax(int n, float *x) {
    float *d_x;
    int *d_ind;
    int max_ind = 0;

    hipMalloc(&d_x, n * sizeof(float));
    hipMalloc(&d_ind, sizeof(int));

    hipMemcpy(d_x, x, n * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_ind, &max_ind, sizeof(int), hipMemcpyHostToDevice);

    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;

    isamax_kernel<<<grid_size, block_size>>>(n, d_x, d_ind);

    hipMemcpy(&max_ind, d_ind, sizeof(int), hipMemcpyDeviceToHost);

    hipFree(d_x);
    hipFree(d_ind);

    return max_ind;
}