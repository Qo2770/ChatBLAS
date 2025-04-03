#include "chatblas_hip.h"

__global__ void sdot_kernel(int n, float *x, float *y, float *res) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    __shared__ float partialSum[256];
    partialSum[threadIdx.x] = 0;

    if(idx < n) {
        partialSum[threadIdx.x] = x[idx] * y[idx];
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float sum = 0;
        for (int i = 0; i < blockDim.x; i++) {
            sum += partialSum[i];
        }
        atomicAdd(res, sum);
    }
}

float chatblas_sdot( int n, float *x, float *y) {
    float *d_x, *d_y, *d_res;
    float res = 0.0f;
    hipMalloc((void**)&d_x, n * sizeof(float));
    hipMalloc((void**)&d_y, n * sizeof(float));
    hipMalloc((void**)&d_res, sizeof(float));
   
    hipMemcpy(d_x, x, n * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_y, y, n * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_res, &res, sizeof(float), hipMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    hipLaunchKernelGGL(sdot_kernel, dim3(gridSize), dim3(blockSize), 0, 0, n, d_x, d_y, d_res);

    hipMemcpy(&res, d_res, sizeof(float), hipMemcpyDeviceToHost);
   
    hipFree(d_x); 
    hipFree(d_y); 
    hipFree(d_res); 

    return res;
}
