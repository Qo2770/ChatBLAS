#include "chatblas_cuda.h"

__global__ void sdsdot_kernel(int n, float b, float *x, float *y, float *res) {
  // Define shared memory for storing partial dot products
  __shared__ double partialSum[256];

  // Define thread index and stride
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  // Perform the dot product calculation in parallel
  double localSum = 0.0;
  for (int i = tid; i < n; i += stride) {
    localSum += (double)x[i] * (double)y[i];
  }

  // Store the partial sum in shared memory
  partialSum[threadIdx.x] = localSum;

  // Synchronize threads within the block
  __syncthreads();

  // Perform reduction on the shared memory
  int i = blockDim.x / 2;
  while (i != 0) {
    if (threadIdx.x < i) {
      partialSum[threadIdx.x] += partialSum[threadIdx.x + i];
    }
    __syncthreads();
    i /= 2;
  }

  // Store the final dot product in global memory
  if (threadIdx.x == 0) {
    res[blockIdx.x] = (float)(partialSum[0] + (double)b);
  }
}

float chatblas_sdsdot(int n, float b, float *x, float *y) {
  // Declare pointers for GPU memory
  float *d_x, *d_y, *d_res;

  // Allocate GPU memory for vectors x, y, and result
  cudaMalloc((void **)&d_x, n * sizeof(float));
  cudaMalloc((void **)&d_y, n * sizeof(float));
  cudaMalloc((void **)&d_res, sizeof(float));

  // Transfer data from CPU to GPU
  cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, n * sizeof(float), cudaMemcpyHostToDevice);

  // Define the number of threads per block and number of blocks
  int threadsPerBlock = 256;
  int numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;

  // Call the kernel function
  sdsdot_kernel<<<numBlocks, threadsPerBlock>>>(n, b, d_x, d_y, d_res);

  // Transfer result back from GPU to CPU
  float result;
  cudaMemcpy(&result, d_res, sizeof(float), cudaMemcpyDeviceToHost);

  // Free GPU memory
  cudaFree(d_x);
  cudaFree(d_y);
  cudaFree(d_res);

  return result;
}
