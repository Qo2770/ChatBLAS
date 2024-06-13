#include "chatblas_hip.h"

__device__ void sdsdot_kernel( int n, float b, float *x, float *y, float *res ) {
  
    __shared__ double cache[blockSize];
    int tid = threadIdx.x;
    int cacheIndex = threadIdx.x;
    double temp = 0;

    while (tid < n) {
        temp += ((double) x[tid]) * ((double) y[tid]);
        tid += blockSize;
    }
    
    cache[cacheIndex] = temp;
    
    __syncthreads();

    int i = blockDim.x / 2;
    while (i != 0) {
        if (cacheIndex < i)
            cache[cacheIndex] += cache[cacheIndex + i];
        __syncthreads();
        i /= 2;
    }
    
    if (cacheIndex == 0)
        res[blockIdx.x] = (float) cache[0];
}

float chatblas_sdsdot( int n, float b, float *x, float *y) {
    float *dx, *dy, *dres, result;

    hipMalloc(&dx, n*sizeof(float));
    hipMalloc(&dy, n*sizeof(float));
    hipMalloc(&dres, blocks*sizeof(float));

    hipMemcpy(dx, x, n*sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(dy, y, n*sizeof(float), hipMemcpyHostToDevice);
       
    sdsdot_kernel <<< blocks, blockSize>>> (n, b, dx, dy, dres);
    
    float *hres = (float *) malloc(blocks*sizeof(float));
    hipMemcpy(hres, dres, blocks*sizeof(float), hipMemcpyDeviceToHost);
    
    for (int i = 0; i < blocks; i++)
        result += hres[i];
    
    hipFree(dx);
    hipFree(dy);
    hipFree(dres);
    free(hres);

    return result + b;
}