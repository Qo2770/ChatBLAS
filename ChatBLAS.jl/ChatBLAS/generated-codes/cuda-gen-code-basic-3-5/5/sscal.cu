#include "chatblas_cuda.h"

__global__
void chatblas_sscal_kernel(int n, float a, float* x) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        x[idx] *= a;
    }
}

void chatblas_sscal(int n, float a, float* x) {
    int block_size = 256;
    int num_blocks = (n + block_size - 1) / block_size;
    
    chatblas_sscal_kernel<<<num_blocks, block_size>>>(n, a, x);
    cudaDeviceSynchronize();
}
