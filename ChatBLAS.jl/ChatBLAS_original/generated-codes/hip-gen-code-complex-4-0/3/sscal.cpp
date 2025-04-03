#include "chatblas_hip.h"

__global__ void sscal_kernel( int n, float a , float *x ) {
   int idx = blockIdx.x * blockDim.x + threadIdx.x;

   if (idx < n) {
       x[idx] = a * x[idx];
   }
}

void chatblas_sscal( int n, float a, float *x) {
   float *dev_x;
   size_t size = n * sizeof(float);

   hipMalloc((void**) &dev_x, size );
   hipMemcpy(dev_x,x,size,hipMemcpyHostToDevice);

   int threadsPerBlock = 256;
   int blocksPerGrid =(n + threadsPerBlock - 1) / threadsPerBlock;
   sscal_kernel<<<blocksPerGrid, threadsPerBlock>>>(n, a, dev_x);

   hipMemcpy(x, dev_x, size, hipMemcpyDeviceToHost);
   hipFree(dev_x);
}