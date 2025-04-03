#include "chatblas_cuda.h"

__global__ void sscal_kernel(int n, float a, float *x) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (idx < n) {
    x[idx] = a * x[idx];
  }
}

void chatblas_sscal(int n, float a, float *x) {
  float *d_x;
  
  // Allocate memory on the GPU
  cudaMalloc((void **)&d_x, n * sizeof(float));
  
  // Transfer data from CPU to GPU
  cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
  
  // Define the number of blocks and threads per block
  int threads_per_block = 256;
  int num_blocks = (n + threads_per_block - 1) / threads_per_block;
  
  // Launch the kernel
  sscal_kernel<<<num_blocks, threads_per_block>>>(n, a, d_x);
  
  // Transfer data from GPU to CPU
  cudaMemcpy(x, d_x, n * sizeof(float), cudaMemcpyDeviceToHost);
  
  // Free memory on the GPU
  cudaFree(d_x);
}
