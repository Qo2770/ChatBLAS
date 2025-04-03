#include "chatblas_cuda.h"

__global__ void isamax_kernel(int n, float *x, float *ind) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    __shared__ float max_val[256];
    __shared__ int max_loc[256];

    max_val[threadIdx.x] = (i < n) ? fabsf(x[i]) : -1;
    max_loc[threadIdx.x] = (i < n) ? i : -1;

    __syncthreads();

    int numActiveThreads = blockDim.x;
    while (numActiveThreads > 1) {
        int halfPoint = (numActiveThreads >> 1);
        if (threadIdx.x < halfPoint) {
            int secondHalfIndex = threadIdx.x + halfPoint;
            if (max_val[threadIdx.x] < max_val[secondHalfIndex]) {
                max_val[threadIdx.x] = max_val[secondHalfIndex];
                max_loc[threadIdx.x] = max_loc[secondHalfIndex];
            }
        }
        __syncthreads();
        numActiveThreads = halfPoint;
    }

    if (threadIdx.x == 0) {
        ind[blockIdx.x] = max_loc[0];
    }
}

int chatblas_isamax(int n, float *x) {
    float *d_x, *d_ind;
    cudaMalloc((void **)&d_x, n * sizeof(float));
    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockDim(256);
    int gridSize = ceil((float)n / blockDim.x);
    cudaMalloc((void **)&d_ind, gridSize * sizeof(float));

    isamax_kernel<<<gridSize, blockDim>>>(n, d_x, d_ind);

    float *indices = (float *)malloc(gridSize * sizeof(float));
    cudaMemcpy(indices, d_ind, gridSize * sizeof(float), cudaMemcpyDeviceToHost);

    int maxIdx = indices[0];
    for (int i = 1; i < gridSize; ++i) {
        if (x[maxIdx] < x[(int)indices[i]]) {
            maxIdx = indices[i];
        }
    }

    cudaFree(d_x);
    cudaFree(d_ind);
    free(indices);
  
    return maxIdx;
}
