#include "chatblas_cuda.h"

__global__ void isamax_kernel(int n, float *x, float *ind) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (tid < n) {
    atomicMin(ind, fabsf(x[tid]));
  }
}

int chatblas_isamax(int n, float *x) {
  float *d_x, *d_ind;
  float h_ind = 0;
  
  // Allocate GPU memory
  cudaMalloc((void**)&d_x, n * sizeof(float));
  cudaMalloc((void**)&d_ind, sizeof(float));
  
  // Copy data from CPU to GPU
  cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
  
  // Set initial values for ind
  cudaMemcpy(d_ind, &h_ind, sizeof(float), cudaMemcpyHostToDevice);
  
  // Define block size and number of blocks
  int block_size = 256;
  int num_blocks = (n + block_size - 1) / block_size;
  
  // Launch CUDA kernel
  isamax_kernel<<<num_blocks, block_size>>>(n, d_x, d_ind);
  
  // Copy data back from GPU to CPU
  cudaMemcpy(&h_ind, d_ind, sizeof(float), cudaMemcpyDeviceToHost);
  
  // Free GPU memory
  cudaFree(d_x);
  cudaFree(d_ind);
  
  return h_ind;
}
