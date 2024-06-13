#include "chatblas_hip.h"

__global__ void sasum_kernel(int n, float *x, float *sum) {
  float *data = SharedMemory<float>();
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  data[tid] = (i < n) ? fabs(x[i]) : 0;
  __syncthreads();

  for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
    if (tid < s) {
      data[tid] += data[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    atomicAdd(sum, data[0]);
  }
}

float chatblas_sasum(int n, float *x) {
  int size = n * sizeof(float);
  float sum = 0.0f;
  float *d_x, *d_sum;
  hipMalloc(&d_x, size);
  hipMalloc(&d_sum, sizeof(float));
  
  hipMemcpy(d_x, x, size, hipMemcpyHostToDevice);
  hipMemset(d_sum, 0, sizeof(float));

  int blockSize = 256;
  int gridSize = (n + blockSize - 1) / blockSize;

  hipLaunchKernelGGL(sasum_kernel, dim3(gridSize), dim3(blockSize), blockSize * sizeof(float), 0, n, d_x, d_sum);
  
  hipMemcpy(&sum, d_sum, sizeof(float), hipMemcpyDeviceToHost);

  hipFree(d_x);
  hipFree(d_sum);

  return sum;
}