#include "chatblas_cuda.h"

__global__ void snrm2_kernel(int n, float *x, float *res) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  float local_sum = 0.0f;
  for (int i = index; i < n; i += stride) {
    local_sum += x[i] * x[i];
  }

  atomicAdd(res, local_sum);
}

float chatblas_snrm2(int n, float *x) {
  float *d_x, *d_res;
  float res = 0.0f;

  cudaMalloc((void**)&d_x, n * sizeof(float));
  cudaMalloc((void**)&d_res, sizeof(float));

  cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemset(d_res, 0, sizeof(float));

  int threadsPerBlock = 256;
  int numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;

  snrm2_kernel<<<numBlocks, threadsPerBlock>>>(n, d_x, d_res);

  cudaMemcpy(&res, d_res, sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_x);
  cudaFree(d_res);

  return sqrt(res);
}
