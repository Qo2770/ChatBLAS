#include "chatblas_cuda.h"

__global__ void isamax_kernel(int n, float *x, int *max_index, float *max_value) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    
    float local_max_value = 0;
    int local_max_index = -1;
    
    for (int i = tid; i < n; i += stride) {
        float abs_val = fabsf(x[i]);
        if (abs_val > local_max_value) {
            local_max_value = abs_val;
            local_max_index = i;
        }
    }
    
    atomicMax(max_value, local_max_value);
    
    if (local_max_value == *max_value)
        atomicMax(max_index, local_max_index);
}

int chatblas_isamax(int n, float *x) {
    const int block_size = 256;
    int num_blocks = (n + block_size - 1) / block_size;
    
    float *max_value_device;
    int *max_index_device;
    
    cudaMalloc((void **)&max_value_device, sizeof(float));
    cudaMalloc((void **)&max_index_device, sizeof(int));
    
    float max_value_host = 0;
    int max_index_host = -1;
    
    cudaMemcpy(max_value_device, &max_value_host, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(max_index_device, &max_index_host, sizeof(int), cudaMemcpyHostToDevice);
    
    isamax_kernel<<<num_blocks, block_size>>>(n, x, max_index_device, max_value_device);
    
    cudaMemcpy(&max_index_host, max_index_device, sizeof(int), cudaMemcpyDeviceToHost);
    
    cudaFree(max_index_device);
    cudaFree(max_value_device);
    
    return max_index_host;
}
