#include "chatblas_hip.h"

__global__ void isamax_kernel(int n, float *x, int *ind) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  __shared__ float smax[256];
  __shared__ int sind[256];

  if (tid < n) {
    smax[threadIdx.x] = abs(x[tid]);
    sind[threadIdx.x] = tid;
  } else {
    smax[threadIdx.x] = 0;
    sind[threadIdx.x] = -1;
  }
  __syncthreads();

  for(unsigned int s=blockDim.x/2; s>0; s>>=1) {
    if (threadIdx.x < s) {
      if (smax[threadIdx.x] < smax[threadIdx.x + s]) {
        smax[threadIdx.x] = smax[threadIdx.x + s];
        sind[threadIdx.x] = sind[threadIdx.x + s];
      }
    }
    __syncthreads();
  }

  if(threadIdx.x == 0) {
    ind[blockIdx.x] = sind[0];
  }
}

int chatblas_isamax(int n, float *x) {
  float *x_d;
  int *ind_d, ind_h;
  hipMalloc((void**)&x_d, n * sizeof(float));
  hipMalloc((void**)&ind_d, sizeof(int));
  hipMemcpy(x_d, x, n * sizeof(float), hipMemcpyHostToDevice);
  isamax_kernel<<<(n + 255) / 256, 256>>>(n, x_d, ind_d);
  hipMemcpy(&ind_h, ind_d, sizeof(int), hipMemcpyDeviceToHost);
  hipFree(x_d);
  hipFree(ind_d);
  return ind_h;
}
