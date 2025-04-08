#include "chatblas_cuda.h"

__global__ void absMaxKernel(int n, float *x, int *maxIdx) {
    extern __shared__ float sharedAbsVal[];
    extern __shared__ int sharedIdx[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    if (idx < n) {
        sharedAbsVal[tid] = fabsf(x[idx]);
        sharedIdx[tid] = idx;
    } else {
        sharedAbsVal[tid] = -1.0f;  // Negative value, since we're dealing with fabsf.
        sharedIdx[tid] = -1;
    }

    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (sharedAbsVal[tid] < sharedAbsVal[tid + stride]) {
                sharedAbsVal[tid] = sharedAbsVal[tid + stride];
                sharedIdx[tid] = sharedIdx[tid + stride];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        maxIdx[blockIdx.x] = sharedIdx[0];
    }
}

int chatblas_isamax(int n, float *x) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    int *d_maxIdx, *h_maxIdx = (int *)malloc(blocksPerGrid * sizeof(int));

    cudaMalloc((void **)&d_maxIdx, blocksPerGrid * sizeof(int));

    absMaxKernel<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(n, x, d_maxIdx);

    cudaMemcpy(h_maxIdx, d_maxIdx, blocksPerGrid * sizeof(int), cudaMemcpyDeviceToHost);

    int maxIndex = -1;
    float maxAbsValue = -1.0f;
    for (int i = 0; i < blocksPerGrid; i++) {
        float currentAbsValue = fabsf(x[h_maxIdx[i]]);
        if (currentAbsValue > maxAbsValue) {
            maxAbsValue = currentAbsValue;
            maxIndex = h_maxIdx[i];
        }
    }

    cudaFree(d_maxIdx);
    free(h_maxIdx);

    return maxIndex;
}
