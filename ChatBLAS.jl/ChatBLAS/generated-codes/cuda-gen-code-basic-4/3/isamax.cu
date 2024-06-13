#include "chatblas_cuda.h"

__global__ void findMaxIndexKernel(float* d_x, int* d_maxIndex, int n)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    float max_val;

    if(i < n)
    {
        if (fabs(d_x[i]) > max_val)
        {
            max_val = fabs(d_x[i]);
            *d_maxIndex = i;
        }
    }
}


int chatblas_isamax(int n, float *x){
   
    float* d_x;
    int* d_maxIndex;
    int maxIndex = 0;

    cudaMalloc((void**)&d_x, n * sizeof(float));
    cudaMalloc((void**)&d_maxIndex, sizeof(int));

    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_maxIndex, &maxIndex, sizeof(int), cudaMemcpyHostToDevice);

    findMaxIndexKernel<<<(n+255)/256, 256>>>(d_x, d_maxIndex, n);

    cudaMemcpy(&maxIndex, d_maxIndex, sizeof(int), cudaMemcpyDeviceToHost);
    
    cudaFree(d_x);
    cudaFree(d_maxIndex);
        
    return maxIndex;
}
