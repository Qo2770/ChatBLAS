#include "chatblas_hip.h"

__global__ 
void sasum_kernel(int n, float *x, float *sum) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    float temp = 0;
    __shared__ float partialSummands[256];
    if(tid < n){
        temp = fabs(x[tid]);
    }
    partialSummands[threadIdx.x] = temp;
    __syncthreads();
    if(tid < n && threadIdx.x == 0)
    {
        for(int i=0; i<blockDim.x; i++){
            *sum += partialSummands[i];
        }
    }
}

float chatblas_sasum(int n, float *x) {
    float *d_x, *d_sum, *h_sum;
    h_sum = (float*)malloc(sizeof(float));
    *h_sum = 0.0f;
    hipMalloc((void**)&d_x, n*sizeof(float));
    hipMalloc((void**)&d_sum, sizeof(float));

    hipMemcpy(d_x, x, n*sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_sum, h_sum, sizeof(float), hipMemcpyHostToDevice);
    dim3 dimBlock(256, 1, 1);
    dim3 dimGrid((n + dimBlock.x - 1) / dimBlock.x, 1, 1);
    hipLaunchKernelGGL(sasum_kernel, dimGrid , dimBlock, 0, 0, n, d_x, d_sum);

    hipMemcpy(h_sum, d_sum, sizeof(float), hipMemcpyDeviceToHost);
    float sum = *h_sum;

    hipFree(d_x);
    hipFree(d_sum);
    free(h_sum);
    return sum;
}
