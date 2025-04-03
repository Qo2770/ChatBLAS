#include "chatblas_cuda.h"

__global__ void snrm2_kernel(int n, float *x, float *res) {
  // Thread ID calculation
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  
  // Initialize variables for local thread sum
  float local_sum = 0.0f;
  
  // Compute partial sum using thread cooperation
  for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
    local_sum += x[i] * x[i];
  }
  
  // Store partial sum in shared memory
  __shared__ float shared_sum[256];
  shared_sum[threadIdx.x] = local_sum;
  
  // Synchronize threads within block
  __syncthreads();
  
  // Compute final sum reduction with thread cooperation
  if (threadIdx.x == 0) {
    float block_sum = 0.0f;
    for (int i = 0; i < blockDim.x; i++) {
      block_sum += shared_sum[i];
    }
    atomicAdd(res, block_sum);
  }
}

float chatblas_snrm2(int n, float *x) {
  // Allocate memory on the GPU for the vector x and the result
  float *gpu_x, *gpu_res;
  cudaMalloc((void**)&gpu_x, n * sizeof(float));
  cudaMalloc((void**)&gpu_res, sizeof(float));

  // Copy vector x from the CPU to the GPU
  cudaMemcpy(gpu_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
  
  // Define the number of threads per block and the number of blocks
  int numThreadsPerBlock = 256;
  int numBlocks = (n + numThreadsPerBlock - 1) / numThreadsPerBlock;
  
  // Launch the kernel
  snrm2_kernel<<<numBlocks, numThreadsPerBlock>>>(n, gpu_x, gpu_res);
  
  // Copy the result from the GPU to the CPU
  float res;
  cudaMemcpy(&res, gpu_res, sizeof(float), cudaMemcpyDeviceToHost);
  
  // Free the GPU memory
  cudaFree(gpu_x);
  cudaFree(gpu_res);
  
  // Return the computed Euclidean norm
  return sqrt(res);
}
