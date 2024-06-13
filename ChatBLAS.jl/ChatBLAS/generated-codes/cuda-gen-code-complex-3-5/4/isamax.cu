#include "chatblas_cuda.h"

__global__ void isamax_kernel(int n, float *x, float *ind) {
  // Get the thread ID
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  
  // Initialize variables
  float max_val = 0.0f;
  int max_idx = 0;
  
  // Find the element with largest absolute value
  for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
    float abs_val = abs(x[i]);
    if (abs_val > max_val) {
      max_val = abs_val;
      max_idx = i;
    }
  }
  
  // Store the result in the global memory
  if (tid == 0)
    *ind = (float)max_idx;
}

int chatblas_isamax(int n, float *x) {
  // Declare device vectors
  float *d_x, *d_ind;
  
  // Allocate memory on the GPU
  cudaMalloc((void**)&d_x, n * sizeof(float));
  cudaMalloc((void**)&d_ind, sizeof(float));
  
  // Copy input vector from host to device
  cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
  
  // Calculate the number of threads and blocks
  int threadsPerBlock = 256;
  int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
  
  // Launch the kernel
  isamax_kernel<<<blocksPerGrid, threadsPerBlock>>>(n, d_x, d_ind);
  
  // Copy the result back from device to host
  float max_idx;
  cudaMemcpy(&max_idx, d_ind, sizeof(float), cudaMemcpyDeviceToHost);
  
  // Free the GPU memory
  cudaFree(d_x);
  cudaFree(d_ind);
  
  // Return the position of the largest absolute value
  return (int)max_idx;
}
