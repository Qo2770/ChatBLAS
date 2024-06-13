#include "chatblas_hip.h"

__global__ void sasum_kernel(int n, float *x, float *sum) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < n) {
    atomicAdd(sum, abs(x[idx]));
  }
}

float chatblas_sasum(int n, float *x) {
  float *device_x, *device_sum;
  float host_sum = 0.0f;
  
  hipMalloc((void**)&device_x, n * sizeof(float));
  hipMalloc((void**)&device_sum, sizeof(float));
  
  hipMemcpy(device_x, x, n * sizeof(float), hipMemcpyHostToDevice);
  hipMemcpy(device_sum, &host_sum, sizeof(float), hipMemcpyHostToDevice);

  dim3 threadsPerBlock(256);
  dim3 numBlocks((n + threadsPerBlock.x - 1) / threadsPerBlock.x);
  
  sasum_kernel<<<numBlocks, threadsPerBlock>>>(n, device_x, device_sum);
  
  hipMemcpy(&host_sum, device_sum, sizeof(float), hipMemcpyDeviceToHost);

  hipFree(device_x);
  hipFree(device_sum);
  
  return host_sum;
}
