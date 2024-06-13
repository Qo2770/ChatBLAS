#include "chatblas_cuda.h"

__global__ void sswap_kernel(int n, float *x, float *y) {
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   if (idx < n) {
      float temp = x[idx];
      x[idx] = y[idx];
      y[idx] = temp;
   }
}

void chatblas_sswap(int n, float *x, float *y) {
   float* dev_x;
   float* dev_y;
   int block_size = 256;
   int grid_size = (n + block_size - 1) / block_size;
   
   cudaMalloc((void**)&dev_x, n * sizeof(float));
   cudaMalloc((void**)&dev_y, n * sizeof(float));

   cudaMemcpy(dev_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
   cudaMemcpy(dev_y, y, n * sizeof(float), cudaMemcpyHostToDevice);

   sswap_kernel<<<grid_size, block_size>>>(n, dev_x, dev_y);
   
   cudaMemcpy(x, dev_x, n * sizeof(float), cudaMemcpyDeviceToHost);
   cudaMemcpy(y, dev_y, n * sizeof(float), cudaMemcpyDeviceToHost);

   cudaFree(dev_x);
   cudaFree(dev_y);
}