#include "chatblas_cuda.h"
 
// CUDA Kernel function to add the absolute values of the elements
__global__ void addAbs(int n, float *x, float *result) {
    int index = threadIdx.x;
    int stride = blockDim.x;
   
    for (int i = index; i < n; i += stride){
        atomicAdd(result, fabsf(x[i]));
    }
}

float chatblas_sasum(int n, float *x) {
    float *d_x, *d_result;
    float result = 0.0;

    // Allocate device memory
    cudaMalloc(&d_x, n*sizeof(float));
    cudaMalloc(&d_result, sizeof(float));

    // Copy vectors from host to device memory
    cudaMemcpy(d_x, x, n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_result, &result, sizeof(float), cudaMemcpyHostToDevice);

    // Launch CUDA Kernel
    addAbs<<<1, 256>>>(n, d_x, d_result);

    // Copy result back to host memory
    cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_result);
    cudaFree(d_x);

    return result;
}
