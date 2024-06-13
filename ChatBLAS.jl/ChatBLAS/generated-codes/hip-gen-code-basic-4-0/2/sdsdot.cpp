#include "chatblas_hip.h"
#include <hip/hip_runtime.h>

__global__ void sdsdot_kernel( int n, float b, float *x, float *y, double *res ) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    double tmp = 0.0;
    if (index < n) {
        tmp = x[index] * y[index];
        atomicAdd(res, tmp);
    }
    if(index == 0)
    	*res += b;
}

float chatblas_sdsdot( int n, float b, float *x, float *y) {
    float *d_x, *d_y;
    double *d_res, res = 0;

    hipMalloc((void**)&d_x, n*sizeof(float));
    hipMalloc((void**)&d_y, n*sizeof(float));
    hipMalloc((void**)&d_res, sizeof(double));

    hipMemcpy(d_x, x, n*sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_y, y, n*sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_res, &res, sizeof(double), hipMemcpyHostToDevice);

    hipLaunchKernelGGL(sdsdot_kernel, dim3((n+255)/256), dim3(256), 0, 0, n, b, d_x, d_y, d_res);

    hipMemcpy(&res, d_res, sizeof(double), hipMemcpyDeviceToHost);

    hipFree(d_x);
    hipFree(d_y);
    hipFree(d_res);

    return (float)res;
}
