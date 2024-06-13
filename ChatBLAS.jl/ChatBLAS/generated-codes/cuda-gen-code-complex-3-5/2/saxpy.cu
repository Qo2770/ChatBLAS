#include "chatblas_cuda.h"

__global__ void saxpy_kernel(int n, float a, float *x, float *y) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
    y[index] = a * x[index] + y[index];
  }
}

void chatblas_saxpy(int n, float a, float *x, float *y) {
  // Allocate GPU memory for x and y
  float *d_x, *d_y;
  cudaMalloc((void**)&d_x, n * sizeof(float));
  cudaMalloc((void**)&d_y, n * sizeof(float));

  // Transfer x and y from CPU to GPU
  cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, n * sizeof(float), cudaMemcpyHostToDevice);

  // Set the number of blocks and threads per block
  int threadsPerBlock = 256;
  int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

  // Launch the kernel
  saxpy_kernel<<<blocksPerGrid, threadsPerBlock>>>(n, a, d_x, d_y);

  // Transfer the result from GPU to CPU
  cudaMemcpy(y, d_y, n * sizeof(float), cudaMemcpyDeviceToHost);

  // Free GPU memory
  cudaFree(d_x);
  cudaFree(d_y);
}