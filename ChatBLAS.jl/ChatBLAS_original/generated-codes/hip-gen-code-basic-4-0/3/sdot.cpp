#include "chatblas_hip.h"

__launch_bounds__(256) 
void sdot_kernel(int n, float *x, float *y, float *res) 
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  __shared__ float sdata[256];

  sdata[threadIdx.x] = (i < n) ? x[i] * y[i] : 0;
  
  __syncthreads();

  for(int s = blockDim.x / 2; s > 0; s >>= 1) 
  {
    if(threadIdx.x < s) 
    {
      sdata[threadIdx.x] += sdata[threadIdx.x + s];
    }
    __syncthreads();
  }
  
  if(threadIdx.x == 0) res[blockIdx.x] = sdata[0];
}

float chatblas_sdot( int n, float *x, float *y) 
{
    hipError_t err;
    float* d_x;
    float* d_y;
    float* d_z;
    float* h_z = new float[256];
    float result = 0.0f;

    err = hipMalloc((void**)&d_x, n * sizeof(float));
    if (err != hipSuccess) return 0;

    err = hipMalloc((void**)&d_y, n * sizeof(float));
    if (err != hipSuccess) return 0;

    err = hipMalloc((void**)&d_z, 256 * sizeof(float));
    if (err != hipSuccess) return 0;

    err = hipMemcpy(d_x, x, n * sizeof(float), hipMemcpyHostToDevice);
    if (err != hipSuccess) return 0;

    err = hipMemcpy(d_y, y, n * sizeof(float), hipMemcpyHostToDevice);
    if (err != hipSuccess) return 0;

    dim3 threadsPerBlock (256);
    dim3 numBlocks ((n + threadsPerBlock.x - 1) / threadsPerBlock.x);

    hipLaunchKernel(sdot_kernel, numBlocks, threadsPerBlock, 0, 0, n, d_x, d_y, d_z);

    err = hipMemcpy(h_z, d_z, 256 * sizeof(float), hipMemcpyDeviceToHost);
    if (err != hipSuccess) return 0;

    for(int i = 0; i < 256; i++) result += h_z[i];

    hipFree(d_x);
    hipFree(d_y);
    hipFree(d_z);

    delete[] h_z;   

    return result;
}
