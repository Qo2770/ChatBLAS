#include "chatblas_hip.h"

__global__ void sdsdot_kernel(int n, float b, float *x, float *y, float *res) {
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  __shared__ double cache[256];

  double temp = 0;
  while(idx < n){
    temp += ((double)x[idx])*((double)y[idx]);
    idx += blockDim.x*gridDim.x;
  }

  cache[threadIdx.x] = temp;

  __syncthreads();

  int i = blockDim.x/2;
  while(i != 0){
    if(threadIdx.x < i){
      cache[threadIdx.x] += cache[threadIdx.x + i];
    }
    __syncthreads();
    i /= 2;
  }

  if(threadIdx.x == 0){
    res[blockIdx.x] = (float)cache[0];
  }
}

float chatblas_sdsdot(int n, float b, float *x, float *y) {
  int blocks = 256;
  int threads = 256;

  float *x_dev, *y_dev, *res_dev;
  float res = 0.0;
  float *res_host = (float*)malloc(blocks*sizeof(float));

  hipMalloc((void**) &x_dev, n*sizeof(float));
  hipMemcpy(x_dev, x, n*sizeof(float), hipMemcpyHostToDevice);
  hipMalloc((void**) &y_dev, n*sizeof(float));
  hipMemcpy(y_dev, y, n*sizeof(float), hipMemcpyHostToDevice);
  hipMalloc((void**) &res_dev, blocks*sizeof(float));

  hipLaunchKernelGGL(sdsdot_kernel, blocks, threads, 0, 0, n, b, x_dev, y_dev, res_dev);

  hipMemcpy(res_host, res_dev, blocks*sizeof(float), hipMemcpyDeviceToHost);

  for(int i = 0; i < blocks; i++){
    res += res_host[i];
  }

  hipFree(x_dev);
  hipFree(y_dev);
  hipFree(res_dev);

  free(res_host);

  return (float)( (double)res + (double)b );
}