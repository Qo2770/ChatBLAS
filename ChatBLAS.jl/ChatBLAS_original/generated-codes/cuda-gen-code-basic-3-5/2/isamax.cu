#include "chatblas_cuda.h"

__global__ void findMaxAbsValue(float* x, int* result, int n) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;
  float maxVal = 0.0;
  int maxIndex = -1;

  for (int i = index; i < n; i += stride) {
    float absVal = fabs(x[i]);
    if (absVal > maxVal) {
      maxVal = absVal;
      maxIndex = i;
    }
  }

  atomicMax(result, maxIndex);
}

int chatblas_isamax(int n, float *x) {
  // Allocate memory on the device
  float *d_x;
  cudaMalloc((void**)&d_x, n * sizeof(float));

  // Copy vector X from host to device
  cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);

  // Allocate memory to store the result on the host
  int result;
  int *d_result;
  cudaMalloc((void**)&d_result, sizeof(int));
  cudaMemset(d_result, -1, sizeof(int));

  // Launch kernel with one block and multiple threads
  int blockSize = 256;
  int numBlocks = (n + blockSize - 1) / blockSize;
  findMaxAbsValue<<<numBlocks, blockSize>>>(d_x, d_result, n);

  // Copy the result from device to host
  cudaMemcpy(&result, d_result, sizeof(int), cudaMemcpyDeviceToHost);

  // Free device memory
  cudaFree(d_x);
  cudaFree(d_result);

  return result;
}
