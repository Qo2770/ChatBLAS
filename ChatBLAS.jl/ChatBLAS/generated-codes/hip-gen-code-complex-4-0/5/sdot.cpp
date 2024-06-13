#include "chatblas_hip.h"

__global__ void sdot_kernel(int n, float *x, float *y, float *res) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    __shared__ float cache[256];

    float temp = 0;
    while(idx < n){
        temp += x[idx] * y[idx];
        idx += blockDim.x * gridDim.x;
    }

    cache[threadIdx.x] = temp;

    __syncthreads();

    int i = blockDim.x / 2;
    while(i != 0){
        if(threadIdx.x < i)
            cache[threadIdx.x] += cache[threadIdx.x + i];
        __syncthreads();
        i /= 2;
    }


    if(threadIdx.x == 0)
        atomicAdd(res, cache[0]);
}

float chatblas_sdot( int n, float *x, float *y) {
    float *dx, *dy, *dres, hres = 0;
    hipMalloc((void**)&dx, n * sizeof(float));
    hipMalloc((void**)&dy, n * sizeof(float));
    hipMalloc((void**)&dres, sizeof(float));

    hipMemcpy(dx, x, n*sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(dy, y, n*sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(dres, &hres, sizeof(float), hipMemcpyHostToDevice);

    sdot_kernel<<<128, 256>>>(n, dx, dy, dres);

    hipMemcpy(&hres, dres, sizeof(float), hipMemcpyDeviceToHost);

    hipFree(dx);
    hipFree(dy);
    hipFree(dres);

    return hres;
}