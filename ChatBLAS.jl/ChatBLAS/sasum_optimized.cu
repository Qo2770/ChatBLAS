#include "chatblas_cuda.h" 
/*
__global__ void sasum_kernel(int n, float *x, float *sum) { 
  int index = blockIdx.x * blockDim.x + threadIdx.x; 
  int stride = blockDim.x * gridDim.x; 
  float local_sum = 0.0f;
  if (index < n )
  { 
  //for (int i = index; i < n; i += stride) { 
    //local_sum += abs(x[i]); 
    local_sum += abs(x[index]); 
  //} 
    atomicAdd(sum, local_sum); 
  }
} 
*/

__global__ void sasum_kernel(int n, float *x, float *sum) { 
  int index = blockIdx.x * blockDim.x + threadIdx.x; 
  int stride = blockDim.x;
  __shared__ float sdata[512];
  float local_sum = 0.0f;
  if (index < n )
  { 
    for (int i = index; i < n; i += stride) { 
      local_sum += abs(x[i]); 
    } 
    __syncthreads();
    sdata[index] = local_sum;
    __syncthreads();
    //atomicAdd(sum, local_sum); 
    if (index <= 256){
      local_sum += abs(sdata[index + 256]);
      sdata[index] = local_sum;
    }
    __syncthreads();
    if (index <= 128){
      local_sum += abs(sdata[index + 128]);
      sdata[index] = local_sum;
    }
    __syncthreads();
    if (index <= 64){
      local_sum += abs(sdata[index + 64]);
      sdata[index] = local_sum;
    }
    __syncthreads();
    if (index <= 32){
      local_sum += abs(sdata[index + 32]);
      sdata[index] = local_sum;
    }
    __syncthreads();
    if (index <= 16){
      local_sum += abs(sdata[index + 16]);
      sdata[index] = local_sum;
    }
    __syncthreads();
    if (index <= 8){
      local_sum += abs(sdata[index + 8]);
      sdata[index] = local_sum;
    }
    __syncthreads();
    if (index <= 4){
      local_sum += abs(sdata[index + 4]);
      sdata[index] = local_sum;
    }
    __syncthreads();
    if (index <= 2){
      local_sum += abs(sdata[index + 2]);
      sdata[index] = local_sum;
    }
    __syncthreads();
    if (index <= 1){
      local_sum += abs(sdata[index + 1]);
      sdata[index] = local_sum;
      sum[0] = local_sum;
    }
    __syncthreads();
  }
} 

float chatblas_sasum(int n, float *x) { 
  float *d_x, *d_sum; 
  float sum = 0.0f; 
  cudaMalloc((void **)&d_x, n * sizeof(float)); 
  cudaMalloc((void **)&d_sum, sizeof(float)); 
  cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice); 
  cudaMemcpy(d_sum, &sum, sizeof(float), cudaMemcpyHostToDevice); 
  //int blockSize = 256; 
  int blockSize = 512; 
  //int numBlocks = (n + blockSize - 1) / blockSize; 
  //sasum_kernel<<<numBlocks, blockSize>>>(n, d_x, d_sum); 
  sasum_kernel<<<1, blockSize>>>(n, d_x, d_sum); 
  cudaMemcpy(&sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost); 
  cudaFree(d_x); 
  cudaFree(d_sum); 
  return sum; 
}
