#include "chatblas_cuda.h"

__global__ void snrm2_kernel(int n, float *x, float *res) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;

  __shared__ float cache[256];  // Assuming block_size is 256
  float temp = 0.0f;

  while (index < n) {
    temp += x[index] * x[index];
    index += stride;
  }

  cache[threadIdx.x] = temp;

  __syncthreads();

  int i = blockDim.x / 2;
  while (i != 0) {
    if (threadIdx.x < i) {
      cache[threadIdx.x] += cache[threadIdx.x + i];
    }
    __syncthreads();
    i /= 2;
  }

  if (threadIdx.x == 0) {
    atomicAdd(res, cache[0]);
  }
}

float chatblas_snrm2(int n, float *x) {
  const int block_size = 256;
  const int num_blocks = (n + block_size - 1) / block_size;

  float *d_x, *d_res;
  float res = 0.0f;

  cudaMalloc((void **)&d_x, n * sizeof(float));
  cudaMalloc((void **)&d_res, sizeof(float));

  cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_res, &res, sizeof(float), cudaMemcpyHostToDevice);

  snrm2_kernel<<<num_blocks, block_size>>>(n, d_x, d_res);

  cudaMemcpy(&res, d_res, sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_x);
  cudaFree(d_res);

  return sqrt(res);
}
