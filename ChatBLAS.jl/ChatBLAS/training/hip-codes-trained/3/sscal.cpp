#include "chatblas_hip.h" 

__global__ void sscal_kernel( int n, float a , float *x ) { int tid = blockIdx.x * blockDim.x + threadIdx.x; if (tid < n) { x[tid] *= a; } } 

void chatblas_sscal( int n, float a, float *x) { float *x_gpu; int blockSize = 256; int numBlocks = (n + blockSize - 1) / blockSize; hipMalloc((void**)&x_gpu, n*sizeof(float)); hipMemcpy(x_gpu, x, n*sizeof(float), hipMemcpyHostToDevice); sscal_kernel<<<numBlocks, blockSize>>>(n, a, x_gpu); hipMemcpy(x, x_gpu, n*sizeof(float), hipMemcpyDeviceToHost); hipFree(x_gpu); }
