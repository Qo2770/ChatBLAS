#include "chatblas_hip.h"

__global__ void sasum_kernel(int n, float *x, float *sum) {
    extern __shared__ float s_data[];

    int tid = threadIdx.x;
    int i = blockIdx.x*blockDim.x + threadIdx.x;

    s_data[tid] = (i < n) ? abs(x[i]) : 0;

    __syncthreads();

    for (int s=blockDim.x/2; s>0; s>>=1) {
        if (tid < s) {
            s_data[tid] += s_data[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) atomicAdd(sum, s_data[0]);
}

float chatblas_sasum(int n, float *x) {
   float *d_x, *d_sum;
   float sum = 0.0;
   
   hipMalloc(&d_x, n*sizeof(float));
   hipMemcpy(d_x, x, n*sizeof(float), hipMemcpyHostToDevice);

   hipMalloc(&d_sum, sizeof(float));
   hipMemcpy(d_sum, &sum, sizeof(float), hipMemcpyHostToDevice);
   
   int blockSize = 256;
   int gridSize = (n + blockSize - 1) / blockSize;

   sasum_kernel<<<gridSize,blockSize, blockSize*sizeof(float)>>>(n, d_x, d_sum); 

   hipMemcpy(&sum, d_sum, sizeof(float), hipMemcpyDeviceToHost);
   
   hipFree(d_x);
   hipFree(d_sum);

   return sum;
}