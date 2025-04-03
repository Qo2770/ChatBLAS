#include "cublas_v2.h"
#include "chatblas_cuda.h"

__global__ void chatblas_sscal(int n, float a, float *x){
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    if(threadId < n){
        x[threadId] *= a;
    }
}

void chatblas_sscal_wrapper(int n, float a, float *x){
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    // Assuming that the vector x is already in the device memory
    chatblas_sscal <<<numBlocks, blockSize>>> (n, a, x);

    // Waiting for all threads to finish
    cudaDeviceSynchronize();
}
