#include "chatblas_cuda.h"

__global__ void cuda_sswap(int n, float *x, float *y){
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    // stride is total number of threads
    int stride = blockDim.x * gridDim.x;
               
    for (int i = index; i < n; i += stride) {
        float temp = x[i];
        x[i] = y[i];
        y[i] = temp;
    }
}
//Host code
void chatblas_sswap(int n, float *x, float *y) {
   // Declare variables for number of threads and blocks
    int threadsPerBlock, blocksPerGrid;
    // Initialize values based on longest vector length for efficiency
    if (n > 1024) {
        threadsPerBlock = 1024;
        blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    } else {
        threadsPerBlock = n;
        blocksPerGrid = 1;
    }
    // Call CUDA kernel function in parallel
    cuda_sswap<<<blocksPerGrid, threadsPerBlock>>>(n, x, y);
    
    // check for any errors
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess)
    printf("\nError: %s\n", cudaGetErrorString(err));
}
