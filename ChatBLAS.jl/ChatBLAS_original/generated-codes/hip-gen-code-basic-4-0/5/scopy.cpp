#include "chatblas_hip.h"

__global__ void scopy_kernel( int n, float *x, float *y ) {
  int idx = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  if (idx < n) {
    y[idx] = x[idx];
  }
}

void chatblas_scopy(int n, float *x, float *y) {
    float *xd, *yd;
    hipMalloc((void**)&xd, n*sizeof(float));
    hipMalloc((void**)&yd, n*sizeof(float));
  
    hipMemcpy(xd, x, n*sizeof(float), hipMemcpyHostToDevice);

    dim3 dimGrid(ceil(n/256.0), 1, 1);
    dim3 dimBlock(256, 1, 1);
    hipLaunchKernelGGL(scopy_kernel, dimGrid, dimBlock, 0, 0, n, xd, yd);
    hipDeviceSynchronize();
  
    hipMemcpy(y, yd, n*sizeof(float), hipMemcpyDeviceToHost);
  
    hipFree(xd);
    hipFree(yd);
}
