#include "hip/hip_runtime.h"
#include "chatblas_hip.h"

__global__ void snrm2_kernel( int n, float *x, float *res) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  __shared__ float temp[N];
  
  if(idx<n)  temp[threadIdx.x] = x[idx]*x[idx];
  
  __syncthreads();
  
  if(idx == 0) {
    float sum = 0.0;
    for(int i = 0; i < blockDim.x; i++) {
      sum += temp[i];
    }
    *res = sqrtf(sum);
  }
}

float chatblas_snrm2(int n, float *x) {
  float *d_x, *d_res, result;
  
  hipMalloc(&d_x, n*sizeof(float));
  hipMalloc(&d_res, sizeof(float));
  
  hipMemcpy(d_x, x, n*sizeof(float), hipMemcpyHostToDevice);

  // <<<grid, block>>> denotes the kernel launch configuration
  snrm2_kernel<<<1,n>>>(n, d_x, d_res);
  
  hipMemcpy(&result, d_res, sizeof(float), hipMemcpyDeviceToHost);
  
  hipFree(d_x);
  hipFree(d_res);

  return result;
}
