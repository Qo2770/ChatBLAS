#include "chatblas_cuda.h"

__global__ void isamax_kernel(int n, float *x, float *ind) {
  // Determine the current thread's index
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  // Declare shared memory to store intermediate results
  extern __shared__ float sdata[];
  
  // Initialize shared memory to store the element values
  sdata[threadIdx.x] = (i < n) ? fabsf(x[i]) : 0.0f;
  
  // Synchronize all threads in the block to ensure all shared memory is populated
  __syncthreads();
  
  // Perform reduction to find the position of the largest absolute value
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      if (sdata[threadIdx.x] < sdata[threadIdx.x + s]) {
        sdata[threadIdx.x] = sdata[threadIdx.x + s];
        ind[blockIdx.x] = ind[blockIdx.x + s];
      }
    }
    // Synchronize all threads to ensure correct data is used in next iteration
    __syncthreads();
  }
  
  // The first thread in the block writes the final result to global memory
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *ind = ind[blockIdx.x];
  }
}

int chatblas_isamax(int n, float *x) {
  // Declare and allocate memory for GPU vectors
  float *d_x, *d_ind;
  size_t size = n * sizeof(float);
  cudaMalloc((void **)&d_x, size);
  cudaMalloc((void **)&d_ind, sizeof(float));
  
  // Copy input vector x from host to GPU
  cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice);
  
  // Define the number of threads per block
  int threadsPerBlock = 256;
  
  // Calculate the number of blocks
  int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
  
  // Determine the shared memory size needed for the kernel
  size_t sharedMemSize = threadsPerBlock * sizeof(float);
  
  // Launch the kernel
  isamax_kernel<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(n, d_x, d_ind);
  
  // Copy the result back from GPU to host
  float result;
  cudaMemcpy(&result, d_ind, sizeof(float), cudaMemcpyDeviceToHost);
  
  // Free GPU memory
  cudaFree(d_x);
  cudaFree(d_ind);
  
  // Return the position of the largest absolute value
  return (int)result;
}
