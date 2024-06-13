__global__ void scopy_kernel( int n, float *x, float *y ) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx < n) {
        y[idx] = x[idx];
    }
}
#include "hip/hip_runtime.h"
#include "hip/hip_vector_types.h"

void chatblas_scopy(int n, float *x, float *y) {
    float *d_x, *d_y;

    hipMalloc((void**)&d_x, n * sizeof(float));
    hipMalloc((void**)&d_y, n * sizeof(float));

    hipMemcpy(d_x, x, n * sizeof(float), hipMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    hipLaunchKernelGGL(scopy_kernel, dim3(numBlocks), dim3(blockSize), 0, 0, n, d_x, d_y);
    
    hipMemcpy(y, d_y, n * sizeof(float), hipMemcpyDeviceToHost);

    hipFree(d_x);
    hipFree(d_y);
}
