#include "chatblas_cuda.h"

__global__ void isamax_kernel(int n, float *x, float *ind) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  
  int maxIndex = 0;
  float maxValue = 0;
  
  for (int i = index; i < n; i += stride) {
    float absValue = fabsf(x[i]);
    if (absValue > maxValue) {
      maxValue = absValue;
      maxIndex = i;
    }
  }
  
  atomicMax(ind, maxIndex);
}

int chatblas_isamax(int n, float *x) {
  float *d_x, *d_ind;
  size_t size = n * sizeof(float);
  
  cudaMalloc((void **)&d_x, size);
  cudaMalloc((void **)&d_ind, sizeof(float));
  
  cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice);
  cudaMemset(d_ind, 0, sizeof(float));
  
  int blockSize = 256;
  int numBlocks = (n + blockSize - 1) / blockSize;
  
  isamax_kernel<<<numBlocks, blockSize>>>(n, d_x, d_ind);
  
  float h_ind;
  cudaMemcpy(&h_ind, d_ind, sizeof(float), cudaMemcpyDeviceToHost);
  
  cudaFree(d_x);
  cudaFree(d_ind);
  
  return (int)h_ind;
}
