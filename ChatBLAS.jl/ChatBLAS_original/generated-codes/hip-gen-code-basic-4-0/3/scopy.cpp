#include "chatblas_hip.h"

__global__ void scopy_kernel( int n, float *x, float *y ) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i < n)
        y[i] = x[i];  
}

void chatblas_scopy(int n, float *x, float *y) {
    float *d_x, *d_y;

    // allocate device memory
    hipMalloc((void **) &d_x, n * sizeof(float));
    hipMalloc((void **) &d_y, n * sizeof(float));
  
    // copy host memory to device
    hipMemcpy(d_x, x, n * sizeof(float), hipMemcpyHostToDevice);

    // launch the kernel
    scopy_kernel<<<(n + 255) / 256, 256>>>(n, d_x, d_y);
  
    // copy device memory to host
    hipMemcpy(y, d_y, n * sizeof(float), hipMemcpyDeviceToHost);

    // free device memory
    hipFree(d_x);
    hipFree(d_y);
}
