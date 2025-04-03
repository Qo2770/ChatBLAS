#include "chatblas_hip.h"

__global__ void sasum_kernel(int n, float *x, float *sum)
{  
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  sum[i] = 0;
  if (i < n)
    sum[i] = fabsf(x[i]);
}

float chatblas_sasum(int n, float *x)
{
  float *dx, *dsum;
  float *hsum;
  float sum = 0;
  int blockSize = 256;
  int gridSize = (n + blockSize - 1) / blockSize;
  hsum = (float*)malloc(gridSize*sizeof(float));

  hipMalloc((void**)&dx, n*sizeof(float));
  hipMalloc((void**)&dsum, gridSize*sizeof(float));
  
  hipMemcpy(dx, x, n*sizeof(float), hipMemcpyHostToDevice);

  hipLaunchKernelGGL(sasum_kernel, dim3(gridSize), dim3(blockSize), 0, 0, n, dx, dsum);
  hipMemcpy(hsum, dsum, gridSize*sizeof(float), hipMemcpyDeviceToHost);

  for (int i = 0; i < gridSize; i++)
     sum += hsum[i];

  hipFree(dx);
  hipFree(dsum);
  free(hsum);
  
  return sum; 
}
