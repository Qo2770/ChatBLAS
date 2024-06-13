#include "chatblas_hip.h" 

__global__ void isamax_kernel(int n, float *x, int *ind) { int maxind = 0; float maxval = 0; for (int i = threadIdx.x; i < n; i += blockDim.x * gridDim.x) { float val = fabsf(x[i]); if (val > maxval) { maxval = val; maxind = i; } } atomicMax(ind, maxind); } 

int chatblas_isamax(int n, float *x) { float *d_x; int *d_ind, maxind; hipMalloc(&d_x, n * sizeof(float)); hipMemcpy(d_x, x, n * sizeof(float), hipMemcpyHostToDevice); hipMalloc(&d_ind, sizeof(int)); hipMemset(d_ind, 0, sizeof(int)); hipLaunchKernelGGL(isamax_kernel, dim3(32), dim3(32), 0, 0, n, d_x, d_ind); hipMemcpy(&maxind, d_ind, sizeof(int), hipMemcpyDeviceToHost); hipFree(d_x); hipFree(d_ind); return maxind; }
