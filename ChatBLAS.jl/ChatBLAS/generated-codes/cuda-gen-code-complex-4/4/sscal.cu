#include "chatblas_cuda.h"

__global__ void sscal_kernel( int n, float a , float *x )
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    x[i] = a * x[i];
}

void chatblas_sscal( int n, float a, float *x) 
{
  float *dx;
  const int blockSize = 256;
  const int numBlocks = (n + blockSize - 1) / blockSize;

  cudaMalloc((void **)&dx, n * sizeof(float));
  cudaMemcpy(dx, x, n * sizeof(float), cudaMemcpyHostToDevice);

  sscal_kernel<<<numBlocks, blockSize>>>(n, a, dx);

  cudaMemcpy(x, dx, n * sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(dx);
}