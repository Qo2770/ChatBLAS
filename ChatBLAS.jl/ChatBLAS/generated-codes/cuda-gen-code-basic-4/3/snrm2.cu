#include "chatblas_cuda.h"

// CUDA kernel function to compute the square of each element in device memory
__global__ void squareElementsKernel(int n, float *x)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n) {
      x[i] = x[i] * x[i];
    }
}

// function to calculate sum of squares in device memory
float sumOfSquaresCuda(int n, float *x)
{
  float sum = 0;
  for (int i = 0; i < n; i++) {
    sum += x[i];
  }
  return sum;
}

float chatblas_snrm2(int n, float *x)
{
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    float *d_x;
    float norm;

    // Allocate memory on the device
    cudaMalloc((void**)&d_x, n*sizeof(float));

    // Copy the data to the device
    cudaMemcpy(d_x, x, n*sizeof(float), cudaMemcpyHostToDevice);

    // Launch the kernel
    squareElementsKernel<<<numBlocks, blockSize>>>(n, d_x);

    // Wait for all threads to finish
    cudaDeviceSynchronize();

    // Copy the squared values back to host memory
    cudaMemcpy(x, d_x, n*sizeof(float), cudaMemcpyDeviceToHost);

    // Calculate the sum of squares on the device
    norm = sqrt(sumOfSquaresCuda(n, x));

    // Free the device memory
    cudaFree(d_x);

    return norm;
}
