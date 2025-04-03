#include "chatblas_cuda.h"

__global__ void findMax(float* x, int* maxPos, int n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? x[i] : 0;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = (fabs(sdata[tid]) > fabs(sdata[tid+s])) ? sdata[tid] : sdata[tid+s];
            maxPos[tid] = (fabs(sdata[tid]) > fabs(sdata[tid+s])) ? maxPos[tid] : maxPos[tid+s];
        }
        
        __syncthreads();
    }

    if (tid == 0) maxPos[blockIdx.x] = maxPos[0];
}

int chatblas_isamax(int n, float *x) {
    float* dx;
    int* dPos;
    int maxPos;

    cudaMalloc(&dx, sizeof(float)*n);
    cudaMemcpy(dx, x, sizeof(float)*n, cudaMemcpyHostToDevice);
    cudaMalloc(&dPos, sizeof(int)*n);
    cudaMemset(dPos, 0, sizeof(int)*n);

    findMax<<<(n+255)/256, 256, 256 * sizeof(float)>>>(dx, dPos, n);

    cudaMemcpy(&maxPos, dPos, sizeof(int), cudaMemcpyDeviceToHost);

    return maxPos;
}
