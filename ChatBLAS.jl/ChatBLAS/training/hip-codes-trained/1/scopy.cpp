#include "chatblas_hip.h" 

__global__ void scopy_kernel( int n, float *x, float *y ) { int i = blockIdx.x *blockDim.x + threadIdx.x; if (i < n) { y[i] = x[i]; } } 

void chatblas_scopy(int n, float *x, float *y) { float *d_x, *d_y; hipMalloc((void**)&d_x, n*sizeof(float)); hipMalloc((void**)&d_y, n*sizeof(float)); int blockSize = 256; int numBlocks = (n + blockSize - 1) / blockSize; hipMemcpy(d_x, x, n*sizeof(float), hipMemcpyHostToDevice); scopy_kernel<<<numBlocks, blockSize>>>(n, d_x, d_y); hipMemcpy(y, d_y, n*sizeof(float), hipMemcpyDeviceToHost); hipFree(d_x); hipFree(d_y); }
