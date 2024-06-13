#include "chatblas_hip.h"

__global__ void scopy_kernel( int n, float *x, float *y ) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < n) {
        y[tid] = x[tid];
    }
}

void chatblas_scopy(int n, float *x, float *y) {
    // Allocate GPU memory
    float *d_x, *d_y;
    hipMalloc((void**)&d_x, n*sizeof(float));
    hipMalloc((void**)&d_y, n*sizeof(float));

    // Define the number of blocks and the number of threads per block
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    // Transfer vector x to GPU
    hipMemcpy(d_x, x, n*sizeof(float), hipMemcpyHostToDevice);

    // Call the kernel
    scopy_kernel<<<blocksPerGrid, threadsPerBlock>>>(n, d_x, d_y);

    // Transfer the result back to the CPU
    hipMemcpy(y, d_y, n*sizeof(float), hipMemcpyDeviceToHost);

    // Free GPU memory
    hipFree(d_x);
    hipFree(d_y);
}