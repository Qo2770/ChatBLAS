#include "chatblas_hip.h"

__global__ void sdsdot_kernel(int n, float b, float *x, float *y, double *res)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    double acc = 0;
    if (idx < n)
        acc = x[idx] * y[idx];
    atomicAdd(res, acc);
    if (idx == 0)
        *res += b;
}

float chatblas_sdsdot(int n, float b, float *x, float *y)
{
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    float* d_x;
    float* d_y;
    double* res;
    double result;

    hipMalloc(&d_x, n * sizeof(float));
    hipMalloc(&d_y, n * sizeof(float));
    hipMalloc(&res, sizeof(double));

    hipMemcpy(d_x, x, n * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_y, y, n * sizeof(float), hipMemcpyHostToDevice);
    hipMemset(res, 0, sizeof(double));

    sdsdot_kernel<<<blocksPerGrid, threadsPerBlock>>>(n, b, d_x, d_y, res);

    hipMemcpy(&result, res, sizeof(double), hipMemcpyDeviceToHost);

    hipFree(d_x);
    hipFree(d_y);
    hipFree(res);

    return (float)(result - b);
}
