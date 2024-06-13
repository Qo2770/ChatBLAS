#include "chatblas_cuda.h"

__global__ void sasum_kernel(int n, float *x, float *sum) {
   int index = blockIdx.x * blockDim.x + threadIdx.x;
   int stride = blockDim.x * gridDim.x;

   float local_sum = 0.0f;
   for (int i = index; i < n; i += stride) {
      local_sum += fabsf(x[i]);
   }

   atomicAdd(sum, local_sum);
}

float chatblas_sasum(int n, float *x) {
   float *d_x, *d_sum;
   float sum = 0.0f;

   cudaMalloc((void **)&d_x, n * sizeof(float));
   cudaMalloc((void **)&d_sum, sizeof(float));

   cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
   cudaMemcpy(d_sum, &sum, sizeof(float), cudaMemcpyHostToDevice);

   int blockSize = 256;
   int numBlocks = (n + blockSize - 1) / blockSize;

   sasum_kernel<<<numBlocks, blockSize>>>(n, d_x, d_sum);

   cudaMemcpy(&sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);

   cudaFree(d_x);
   cudaFree(d_sum);

   return sum;
}
