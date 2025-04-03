#include "chatblas_hip.h"

__global__ void sdot_kernel(int n, float *x, float *y, float *res)
{
    __shared__ float cache[512];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;
	
    float temp = 0;
    while (tid < n) {
      temp += x[tid] * y[tid];
      tid += blockDim.x * gridDim.x;
    }
  
    cache[cacheIndex] = temp;
  
    __syncthreads();

    int i = blockDim.x / 2;
    while (i != 0) {
      if (cacheIndex < i)
        cache[cacheIndex] += cache[cacheIndex + i];
      __syncthreads();
      i /= 2;
    }
  
    if (cacheIndex == 0)
      res[blockIdx.x] = cache[0];
}

float chatblas_sdot(int n, float *x, float *y)
{
    float *x_device, *y_device, *res_device, result = 0;
    float *partial_res = (float*) malloc(512 * sizeof(float));

    hipMalloc((void**)&x_device, n*sizeof(float));
    hipMalloc((void**)&y_device, n*sizeof(float));
    hipMalloc((void**)&res_device, 512*sizeof(float));

    hipMemcpy(x_device, x, n*sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(y_device, y, n*sizeof(float), hipMemcpyHostToDevice);
  
    sdot_kernel<<<512, 512>>>(n, x_device, y_device, res_device);

    hipMemcpy(partial_res, res_device, 512*sizeof(float), hipMemcpyDeviceToHost);

    for (int i = 0; i < 512; i++) {
        result += partial_res[i];
    }

    hipFree(x_device);
    hipFree(y_device);
    hipFree(res_device);
    free(partial_res);

    return result;
}