#include "chatblas_cuda.h"

__global__ void sdsdot_kernel( int n, float b, float *x, float *y, float *res ) { 
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  double temp_sum = 0.0;
  if (i < n) {
    temp_sum += ((double)x[i]) * ((double)y[i]);
  }
  atomicAdd(res, (float)temp_sum);
} 

float chatblas_sdsdot( int n, float b, float *x, float *y) {
  float *x_gpu, *y_gpu, *res_gpu; 
  float res = b;

  int blockSize = 256;
  int gridSize = (int)ceil((float)n/blockSize);

  cudaMalloc((void**)&x_gpu, n * sizeof(float));
  cudaMalloc((void**)&y_gpu, n * sizeof(float));
  cudaMalloc((void**)&res_gpu, sizeof(float));

  cudaMemcpy(x_gpu, x, n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(y_gpu, y, n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(res_gpu, &res, sizeof(float), cudaMemcpyHostToDevice);

  sdsdot_kernel<<<gridSize, blockSize>>>(n, b, x_gpu, y_gpu, res_gpu);
  
  cudaMemcpy(&res, res_gpu, sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(x_gpu); 
  cudaFree(y_gpu); 
  cudaFree(res_gpu);

  return res;
}