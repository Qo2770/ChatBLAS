#include "chatblas_cuda.h"

__global__ void sscal_kernel(int n, float a, float *x) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n) {
    x[tid] *= a;
  }
}

void chatblas_sscal(int n, float a, float *x) {
  float *d_x;
  int size = n * sizeof(float);

  // allocate memory on GPU
  cudaMalloc((void **)&d_x, size);

  // copy input data from CPU to GPU
  cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice);

  // define the number of blocks of threads and the size of each block
  int threadsPerBlock = 256;
  int numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;

  // call the kernel function
  sscal_kernel<<<numBlocks, threadsPerBlock>>>(n, a, d_x);

  // copy the result back from GPU to CPU
  cudaMemcpy(x, d_x, size, cudaMemcpyDeviceToHost);

  // free the GPU memory
  cudaFree(d_x);
}
