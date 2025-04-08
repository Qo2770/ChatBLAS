#include "chatblas_cuda.h"

__global__ void findMaxAbsIndex(int n, float *x, int *result) {
    extern __shared__ int sharedIndices[];
    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + tid;

    sharedIndices[tid] = index < n ? tid : -1;

    __syncthreads();

    float maxVal = 0.0f;

    if (index < n) {
        maxVal = fabsf(x[index]);
    }

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        __syncthreads();
        if (tid < stride) {
            float tempVal = fabsf(x[blockIdx.x * blockDim.x + sharedIndices[tid + stride]]);
            if (tempVal > maxVal) {
                maxVal = tempVal;
                sharedIndices[tid] = sharedIndices[tid + stride];
            }
        }
    }

    if (tid == 0) {
        result[blockIdx.x] = sharedIndices[0];
    }
}

int chatblas_isamax(int n, float *x) {
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    float *d_x;
    int *d_result;
    int *h_result = (int *)malloc(numBlocks * sizeof(int));

    cudaMalloc((void **)&d_x, n * sizeof(float));
    cudaMalloc((void **)&d_result, numBlocks * sizeof(int));

    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);

    findMaxAbsIndex<<<numBlocks, blockSize, blockSize * sizeof(int)>>>(n, d_x, d_result);

    cudaMemcpy(h_result, d_result, numBlocks * sizeof(int), cudaMemcpyDeviceToHost);

    int maxIndex = h_result[0];
    for (int i = 1; i < numBlocks; i++) {
        if (fabsf(x[h_result[i]]) > fabsf(x[maxIndex])) {
            maxIndex = h_result[i];
        }
    }

    cudaFree(d_x);
    cudaFree(d_result);
    free(h_result);

    return maxIndex;
}
