#include "chatblas_hip.h"

#define BLOCKSIZE 256

__global__ void snrm2_kernel(int n, float *x, float *res)
{
  __shared__ float sdata[BLOCKSIZE];

  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  sdata[tid] = (i < n) ? x[i]*x[i] : 0.0f;

  __syncthreads();

  for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
    if (tid < s)
      sdata[tid] += sdata[tid + s];
    __syncthreads();
  } 
  
  if (tid == 0 )
    atomicAdd(res, sqrt(sdata[0]));
}

float chatblas_snrm2(int n, float* x)
{
  float *d_x=0, *d_res=0, res=0;

  hipMalloc(&d_x, n * sizeof(float));
  hipMemcpy(d_x, x, n * sizeof(float), hipMemcpyHostToDevice);
  
  hipMalloc(&d_res, sizeof(float));
  hipMemset(d_res, 0, sizeof(float));

  snrm2_kernel <<< (n + BLOCKSIZE - 1) / BLOCKSIZE, BLOCKSIZE >>> (n, d_x, d_res);
  hipDeviceSynchronize();

  hipMemcpy(&res, d_res, sizeof(float), hipMemcpyDeviceToHost);

  hipFree(d_x);
  hipFree(d_res);

  return res;
}
