#include "chatblas_cuda.h"
#include <math.h>
__global__ void sum_abs_kernel(float* x, float* y, unsigned int n) {
    unsigned int index = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;

    __shared__ float cache[256];

    float temp = 0;
    while(index < n) {
        temp += fabs(x[index]);
        index += stride;
    }

    cache[threadIdx.x] = temp;

    __syncthreads();

    if(threadIdx.x == 0) {
        float sum = 0;
        for(int i = 0; i < blockDim.x; i++)
            sum += cache[i];
        
        atomicAdd(y, sum);
    }
}

float chatblas_sasum(int n, float *x) {
    float *dev_x, *dev_y;
    float result;

    cudaMalloc((void**)&dev_x, n*sizeof(float));
    cudaMalloc((void**)&dev_y, sizeof(float));

    cudaMemcpy(dev_x, x, n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(dev_y, 0, sizeof(float));

    sum_abs_kernel<<<256,256>>>(dev_x, dev_y, n);

    cudaMemcpy(&result, dev_y, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dev_x);
    cudaFree(dev_y);

    return result;
}
