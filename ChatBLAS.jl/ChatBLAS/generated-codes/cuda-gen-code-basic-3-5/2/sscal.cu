#include "chatblas_cuda.h"

__global__
void scalar_scale(float a, float* x, int n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n) {
    x[tid] *= a;
  }
}

void chatblas_sscal(int n, float a, float* x) {
  float* dev_x;

  // Allocate memory on the device for x
  cudaMalloc((void**)&dev_x, n * sizeof(float));

  // Copy vector x from host to device
  cudaMemcpy(dev_x, x, n * sizeof(float), cudaMemcpyHostToDevice);

  // Calculate the number of threads per block
  int threadsPerBlock = 256;
  int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

  // Launch the kernel function
  scalar_scale<<<blocksPerGrid, threadsPerBlock>>>(a, dev_x, n);

  // Copy the result back from the device to the host
  cudaMemcpy(x, dev_x, n * sizeof(float), cudaMemcpyDeviceToHost);

  // Free the allocated memory on the device
  cudaFree(dev_x);
}
