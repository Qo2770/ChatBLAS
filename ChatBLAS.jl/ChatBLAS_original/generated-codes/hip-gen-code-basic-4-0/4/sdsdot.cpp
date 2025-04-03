#include "chatblas_hip.h"

__global__ void sdsdot_kernel( int n, float b, float *x, float *y, double *res ) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx<n) {
        res[idx] = (double)x[idx] * (double)y[idx];
    }
    __syncthreads();

    if(idx == 0) {
        double sum = b;
        for(int i=0; i<n; i++) {
            sum += res[i];
        }
        res[0] = sum;
    }
}

float chatblas_sdsdot( int n, float b, float *x, float *y) {
    double *d_x, *d_y, *d_res;
    double res = 0;
    int size = n * sizeof(float);
    int blockSize = 1024;
    int gridSize = (int)ceilf(n / blockSize);

    hipMalloc((void**)&d_x, size);
    hipMalloc((void**)&d_y, size);
    hipMalloc((void**)&d_res, size);

    hipMemcpy(d_x, x, size, hipMemcpyHostToDevice);
    hipMemcpy(d_y, y, size, hipMemcpyHostToDevice);

    sdsdot_kernel<<<gridSize, blockSize>>>(n, b, d_x, d_y, d_res);

    hipMemcpy(&res, d_res, sizeof(double), hipMemcpyDeviceToHost);

    hipFree(d_x);
    hipFree(d_y);
    hipFree(d_res);

    return (float)res;
}
