#include "chatblas_cuda.h"

__global__ void isamax_kernel(int n, float *x, float *ind) {
    // Calculate the global thread ID
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Initialize the variables for maximum value and its position
    float max_val = 0;
    int max_pos = 0;
    
    // Find the position of the element with the largest absolute value
    for (int i = index; i < n; i += blockDim.x * gridDim.x) {
        float val = fabsf(x[i]);
        if (val > max_val) {
            max_val = val;
            max_pos = i;
        }
    }
    
    // Write the result back to global memory
    if (index == 0) {
        *ind = max_pos;
    }
}

int chatblas_isamax(int n, float *x) {
    int block_size = 256;
    int num_blocks = (n + block_size - 1) / block_size;
    
    float *d_x, *d_ind;
    
    // Allocate GPU memory for input vector
    cudaMalloc(&d_x, n * sizeof(float));
    
    // Allocate GPU memory for output index
    cudaMalloc(&d_ind, sizeof(float));
    
    // Copy input vector from CPU to GPU
    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
    
    // Launch the kernel
    isamax_kernel<<<num_blocks, block_size>>>(n, d_x, d_ind);
    
    // Copy the result index from GPU to CPU
    float h_ind;
    cudaMemcpy(&h_ind, d_ind, sizeof(float), cudaMemcpyDeviceToHost);
    
    // Free GPU memory
    cudaFree(d_x);
    cudaFree(d_ind);
    
    return (int)h_ind;
}
