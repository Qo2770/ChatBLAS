#include "chatblas_hip.h"

__global__ void isamax_kernel(int n, float *x, int *index) {
    extern __shared__ float sm_data[];

    int index_in_block = threadIdx.x;
    int global_index = blockIdx.x * blockDim.x + threadIdx.x;

    float max_value_in_block = 0.0;
    int max_index_in_block = -1;

    if (global_index < n) {
        max_value_in_block = fabs(x[global_index]);
        max_index_in_block = global_index;
    }

    sm_data[index_in_block] = max_value_in_block;
    sm_data[index_in_block + blockDim.x] = max_index_in_block;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (index_in_block < offset && sm_data[index_in_block + offset] > sm_data[index_in_block]) {
            sm_data[index_in_block] = sm_data[index_in_block + offset];
            sm_data[index_in_block + blockDim.x] = sm_data[index_in_block + offset + blockDim.x];
        }
        __syncthreads();
    }

    if (index_in_block == 0) {
        index[blockIdx.x] = sm_data[index_in_block + blockDim.x];
    }
}

int chatblas_isamax(int n, float *x) {
    int threadsBlockSize = 256;
    int blocksGridSize = (n + threadsBlockSize - 1) / threadsBlockSize;

    float* dev_x;
    int* dev_ind;
    int* ind = new int[blocksGridSize];

    hipMalloc((void**)&dev_x, n * sizeof(float));
    hipMalloc((void**)&dev_ind, blocksGridSize * sizeof(int));

    hipMemcpy(dev_x, x, n * sizeof(float), hipMemcpyHostToDevice);

    hipLaunchKernelGGL(isamax_kernel, dim3(blocksGridSize), dim3(threadsBlockSize), threadsBlockSize * 2 * sizeof(float), 0, n, dev_x, dev_ind);

    hipMemcpy(ind, dev_ind, blocksGridSize * sizeof(int), hipMemcpyDeviceToHost);

    int index = -1;
    float max = 0;
    for (int i = 0; i < blocksGridSize; i++) {
        if (fabs(x[ind[i]]) > max) {
            max = fabs(x[ind[i]]);
            index = ind[i];
        }
    }

    hipFree(dev_x);
    hipFree(dev_ind);

    delete[] ind;

    return index;
}