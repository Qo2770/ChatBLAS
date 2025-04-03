#include "chatblas_hip.h"

__global__ void isamax_kernel(int n, float *x, int *ind) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ float maxvals[256];
    __shared__ int maxinds[256];

    if (i < n) {
        maxvals[threadIdx.x] = fabsf(x[i]);
        maxinds[threadIdx.x] = i;
    }

    __syncthreads();

    for (int stride = blockDim.x/2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride && i + stride < n) {
            if (maxvals[threadIdx.x] < maxvals[threadIdx.x + stride]) {
                maxvals[threadIdx.x] = maxvals[threadIdx.x + stride];
                maxinds[threadIdx.x] = maxinds[threadIdx.x + stride];
            }
        }

        __syncthreads();
    }

    if (threadIdx.x == 0) ind[blockIdx.x] = maxinds[0];
}

int chatblas_isamax(int n, float *x) {
    float *d_x;
    int *d_ind, *h_ind, blocks, threads = 256;

    h_ind = (int *)malloc(sizeof(int) * n);
    hipMalloc(&d_x, n*sizeof(float));
    hipMalloc(&d_ind, n*sizeof(int));

    hipMemcpy(d_x, x, n*sizeof(float), hipMemcpyHostToDevice);

    blocks = (n + threads - 1) / threads;

    dim3 dimGrid(blocks);
    dim3 dimBlock(threads);

    isamax_kernel<<<dimGrid,dimBlock>>>(n, d_x, d_ind);

    hipMemcpy(h_ind, d_ind, sizeof(int)*blocks, hipMemcpyDeviceToHost);

    int maxind = h_ind[0];
    for (int i = 1; i < blocks; i++)
        if (x[maxind] < x[h_ind[i]]) maxind = h_ind[i];

    hipFree(d_x);
    hipFree(d_ind);
    free(h_ind);

    return maxind;
}