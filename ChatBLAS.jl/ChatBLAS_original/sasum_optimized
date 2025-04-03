#include "chatblas_hip.h"

__global__ void sasum_kernel(int n, float *x, float *res) {
    __shared__ float cache[512];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;
	
    float temp = 0;
    while (tid < n) {
      //temp += x[tid] * y[tid];
      temp += fabsf(x[tid]); 
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

float chatblas_sasum(int n, float *x) {
    float *d_x;
    float *res_device;
    float result = 0.0f;
    
    float *partial_res = (float*) malloc(512 * sizeof(float));

    hipMalloc(&d_x, n * sizeof(float));
    hipMalloc((void**)&res_device, 512*sizeof(float));

    hipMemcpy(d_x, x, n * sizeof(float), hipMemcpyHostToDevice);
    
    sasum_kernel<<<512, 512>>>(n, d_x, res_device);

    hipMemcpy(partial_res, res_device, 512*sizeof(float), hipMemcpyDeviceToHost);

    for (int i = 0; i < 512; i++) {
        result += partial_res[i];
    }

    hipFree(d_x);
    hipFree(res_device);
    free(partial_res);

    return result;
}
