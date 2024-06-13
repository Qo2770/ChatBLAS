#include "chatblas_hip.h"

__global__ void scopy_kernel( int n, float *x, float *y ) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        y[tid] = x[tid];
    }
}

void chatblas_scopy(int n, float *x, float *y) {
    float *d_x, *d_y;
    int size = n * sizeof(float);

    hipMalloc((void**)&d_x, size);
    hipMalloc((void**)&d_y, size);

    hipMemcpy(d_x, x, size, hipMemcpyHostToDevice);

    dim3 blocks(n/256 + 1, 1, 1);
    dim3 threads(256, 1, 1);

    scopy_kernel<<<blocks, threads>>>(n, d_x, d_y);

    hipMemcpy(y, d_y, size, hipMemcpyDeviceToHost);

    hipFree(d_x);
    hipFree(d_y);
}