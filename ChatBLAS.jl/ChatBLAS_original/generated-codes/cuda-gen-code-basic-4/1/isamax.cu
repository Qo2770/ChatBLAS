#include "chatblas_cuda.h"

#define MAXIMUM_THREADS_PER_BLOCK 1024

__global__ 
void find_max_abs_index(float* array, int n, int* result) {
    __shared__ int shared_index[MAXIMUM_THREADS_PER_BLOCK];
    __shared__ float shared_value[MAXIMUM_THREADS_PER_BLOCK];

    int tid = threadIdx.x;
    int local_index = tid + blockIdx.x * blockDim.x;
    int step_size = blockDim.x * gridDim.x;

    float max_value = 0;
    float max_abs_value = 0;
    int max_abs_index = -1;
    while (local_index < n) {
        float value = array[local_index];
        if (abs(value) > max_abs_value) {
            max_abs_value = abs(value);
            max_abs_index = local_index;
        }
        local_index += step_size;
    }
    shared_index[tid] = max_abs_index;
    shared_value[tid] = max_abs_value;

    __syncthreads();

    for (int i = blockDim.x / 2; i > 0; i /= 2) {
        if (tid < i) {
            int other_index = tid + i;
            if (abs(shared_value[other_index]) > abs(shared_value[tid])) {
                shared_value[tid] = shared_value[other_index];
                shared_index[tid] = shared_index[other_index];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        result[blockIdx.x] = shared_index[0];
    }
}

int chatblas_isamax(int n, float* x) {
    int num_blocks = (n + MAXIMUM_THREADS_PER_BLOCK - 1) / MAXIMUM_THREADS_PER_BLOCK;
    int *dev_result, *host_result;
    host_result = (int*)malloc(num_blocks * sizeof(int));
    cudaMalloc(&dev_result, num_blocks * sizeof(int));

    find_max_abs_index<<<num_blocks, MAXIMUM_THREADS_PER_BLOCK>>>(x, n, dev_result);

    cudaMemcpy(host_result, dev_result, num_blocks * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(dev_result);

    int max_index = host_result[0];
    for(int i = 1; i < num_blocks; i++) {
        if(abs(x[host_result[i]]) > abs(x[max_index])) {
            max_index = host_result[i];
        }
    }

    free(host_result);
    return max_index; 
}
