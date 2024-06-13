#include "chatblas_hip.h"

__global__ void snrm2_kernel(int n, float *x, float *res) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    extern __shared__ float sum[];
    sum[idx] = 0.0;

    if(idx < n){
        sum[idx] = x[idx] * x[idx];
    }
    __syncthreads();

    for(int i = blockDim.x/2; i>0; i >>= 1){
        if (idx < i){
            sum[idx] += sum[idx + i];
        }
        __syncthreads();
    }

    if(idx == 0){
        *res = sqrt(sum[0]);
    }
}

float chatblas_snrm2(int n, float *x) {
    float *x_GPU, *result_GPU, result;
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    hipMalloc((void**)&x_GPU, n * sizeof(float));
    hipMemcpy(x_GPU, x, n * sizeof(float), hipMemcpyHostToDevice);

    hipMalloc((void**)&result_GPU, sizeof(float));
    hipMemset(result_GPU, 0, sizeof(float));

    snrm2_kernel<<<numBlocks, blockSize>>>(n, x_GPU, result_GPU);
    
    hipMemcpy(&result, result_GPU, sizeof(float), hipMemcpyDeviceToHost);

    hipFree(x_GPU);
    hipFree(result_GPU);

    return result;
}