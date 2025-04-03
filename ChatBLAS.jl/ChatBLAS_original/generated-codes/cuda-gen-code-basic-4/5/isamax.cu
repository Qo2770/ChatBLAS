#include "chatblas_cuda.h"
#include <cublas_v2.h>

__global__ void abs_max_index_kernel(float* input, int* output, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < n && abs(input[idx]) > abs(input[*output]))
        atomicExch(output, idx);
}

int chatblas_isamax(int n, float *x) {
    float *dev_x;
    int *dev_output;
    
    cudaMalloc((void** ) &dev_x, n * sizeof(float));
    cudaMemcpy(dev_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
    
    // initial index
    int initial = 0;
    cudaMalloc((void** ) &dev_output, sizeof(int));
    cudaMemcpy(dev_output, &initial, sizeof(int), cudaMemcpyHostToDevice);
    
    // launch the kernel
    int threadsPerBlock = 256;
    int blocksPerGrid =(n + threadsPerBlock - 1) / threadsPerBlock;
    abs_max_index_kernel<<<blocksPerGrid, threadsPerBlock>>>(dev_x, dev_output, n);

    int output;
    cudaMemcpy(&output, dev_output, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(dev_x);
    cudaFree(dev_output);
    
    return output;
}
