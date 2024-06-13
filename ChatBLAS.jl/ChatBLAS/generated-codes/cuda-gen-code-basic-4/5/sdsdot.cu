#include "chatblas_cuda.h"

__global__ void dot_product_kernel(int n, float *x, float *y, float *result) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    float value = 0.0;
    
    if (index < n) {
        value = ((double)x[index]) * ((double)y[index]);
    }
    
    atomicAdd(result, value);
}

float chatblas_sdsdot(int n, float b, float *x, float *y) {
    float *dev_x, *dev_y, *dev_result;
    float result;

    cudaMalloc((void**)&dev_x, n*sizeof(float));
    cudaMalloc((void**)&dev_y, n*sizeof(float));
    cudaMalloc((void**)&dev_result, sizeof(float));

    cudaMemcpy(dev_x, x, n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_y, y, n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_result, &b, sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid =(n + threadsPerBlock - 1) / threadsPerBlock;

    dot_product_kernel<<<blocksPerGrid, threadsPerBlock>>>(n, dev_x, dev_y, dev_result);

    cudaMemcpy(&result, dev_result, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dev_x);
    cudaFree(dev_y);
    cudaFree(dev_result);

    return result;
}
