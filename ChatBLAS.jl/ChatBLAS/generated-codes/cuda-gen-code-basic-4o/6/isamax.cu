#include "chatblas_cuda.h"

__global__ void kernel_iamax(int n, float *x, int *index) {
    extern __shared__ float shared_data[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int gridSize = blockDim.x * gridDim.x;

    float maxVal = 0.0f;
    int maxIdx = -1;

    // In case there are more blocks than there are elements in the vector
    while (i < n) {
        float absVal = fabsf(x[i]);
        if (absVal > maxVal) {
            maxVal = absVal;
            maxIdx = i;
        }
        i += gridSize;
    }

    shared_data[tid] = maxVal;
    __shared__ int shared_indices[blockDim.x];
    shared_indices[tid] = maxIdx;
    __syncthreads();

    // Parallel reduction to find max value and its index
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            if (shared_data[tid] < shared_data[tid + stride]) {
                shared_data[tid] = shared_data[tid + stride];
                shared_indices[tid] = shared_indices[tid + stride];
            }
        }
        __syncthreads();
    }

    // The first thread in the block writes the result to global memory
    if (tid == 0) {
        index[blockIdx.x] = shared_indices[0];
    }
}

int chatblas_isamax(int n, float *x) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    int *d_index, *h_index;
    float *d_x;
    
    cudaMalloc((void **)&d_x, n * sizeof(float));
    cudaMalloc((void **)&d_index, blocksPerGrid * sizeof(int));
    h_index = (int *)malloc(blocksPerGrid * sizeof(int));

    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);

    kernel_iamax<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(n, d_x, d_index);

    cudaMemcpy(h_index, d_index, blocksPerGrid * sizeof(int), cudaMemcpyDeviceToHost);

    // Serial reduction to find the index among the block results
    int finalIndex = h_index[0];
    float maxVal = fabsf(x[finalIndex]);

    for (int i = 1; i < blocksPerGrid; ++i) {
        if (fabsf(x[h_index[i]]) > maxVal) {
            finalIndex = h_index[i];
            maxVal = fabsf(x[h_index[i]]);
        }
    }

    cudaFree(d_x);
    cudaFree(d_index);
    free(h_index);

    return finalIndex;
}
