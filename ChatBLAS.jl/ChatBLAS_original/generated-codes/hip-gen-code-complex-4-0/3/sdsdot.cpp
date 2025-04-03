#include "chatblas_hip.h"

__global__ void sdsdot_kernel( int n, float b, float *x, float *y, float *res ) 
{
  // Shared memory allocation for thread block
  __shared__ double buf[ BLOCK_SIZE ];

  // Index for each thread
  int index = threadIdx.x + blockIdx.x * blockDim.x;

  // Initialize in-block sum at start
  double inBlockSum = 0.0;

  // Thread sum up the products
  while (index < n) 
  {
    inBlockSum += ((double)x[index]) * ((double)y[index]);
    index += blockDim.x * gridDim.x;
  }

  // Store in-block sum in shared memory
  buf[threadIdx.x] = inBlockSum;

  // Synchronize in-block summation
  __syncthreads();

  // Reduction in shared memory
  for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
  {
    if (threadIdx.x < stride)
      buf[threadIdx.x] += buf[threadIdx.x + stride];
    __syncthreads();
  }
  
  // Thread 0 writes the result of the block in the result vector
  if (threadIdx.x == 0)
    atomicAdd(res, (float)buf[0]);
}

float chatblas_sdsdot( int n, float b, float *x, float *y) 
{
  float *d_x = NULL, *d_y = NULL, *d_res = NULL;
  float res = 0.0;
  size_t size = n * sizeof(float);

  // Allocate device memory
  hipMalloc(&d_x, size);
  hipMalloc(&d_y, size);
  hipMalloc(&d_res, sizeof(float));

  // Copy vectors from host to device memory
  hipMemcpy(d_x, x, size, hipMemcpyHostToDevice);
  hipMemcpy(d_y, y, size, hipMemcpyHostToDevice);

  // Kernel launch parameters
  int threadsPerBlock = 256;
  int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

  // Initialize the result on device memory
  hipMemcpy(d_res, &res, sizeof(float), hipMemcpyHostToDevice);

  // Execute the kernel
  hipLaunchKernelGGL((sdsdot_kernel), dim3(blocksPerGrid), dim3(threadsPerBlock), 0, 0, n, b, d_x, d_y, d_res);
  
  // Transfer the result vector from device to host
  hipMemcpy(&res, d_res, sizeof(float), hipMemcpyDeviceToHost);

  // Free device memory
  hipFree(d_x);
  hipFree(d_y);
  hipFree(d_res);

  // Return the dot product of x and y plus b
  return res + b;
}