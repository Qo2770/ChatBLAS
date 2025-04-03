#include "chatblas_cuda.h"

__global__ void kernel_sdsdot(int n, float *x, float *y, float *z) {
   int index = threadIdx.x + blockDim.x * blockIdx.x;
   int stride = blockDim.x * gridDim.x;
   for (int i = index; i < n; i += stride)
       z[i] = ((double) x[i]) * ((double) y[i]);
}

extern "C"
float chatblas_sdsdot(int n, float b, float *x, float *y) {
   int size = n * sizeof(float);
   float *d_x, *d_y, *d_z, result = 0.0, *partial_sums;
   
   cudaMalloc(&d_x, size);
   cudaMalloc(&d_y, size);
   cudaMalloc(&d_z, size);
   cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice);
   cudaMemcpy(d_y, y, size, cudaMemcpyHostToDevice);
   
   int threads_per_block = 256;
   int blocks_per_grid =(n + threads_per_block - 1) / threads_per_block;

   kernel_sdsdot<<<blocks_per_grid, threads_per_block>>>(n, d_x, d_y, d_z);

   partial_sums = (float*)malloc(blocks_per_grid*sizeof(float));
   cudaMemcpy(partial_sums, d_z, blocks_per_grid*sizeof(float), cudaMemcpyDeviceToHost);

   for(int i = 0; i < blocks_per_grid ; i++)
       result += partial_sums[i];
   
   cudaFree(d_x);
   cudaFree(d_y);
   cudaFree(d_z);
   free(partial_sums);
   
   result += b;
   return result;
}
