#include "chatblas_cuda.h"

__global__ void dotproduct(int n, float *x, float *result)
{
    __shared__ float cache[256];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    float temp = 0;
    while(tid < n)
    {
        temp += x[tid]*x[tid];
        tid += blockDim.x*gridDim.x;
    }
    cache[threadIdx.x] = temp;
    __syncthreads();

    // reduction
    int i = blockDim.x/2;
    while(i != 0) {
        if(threadIdx.x < i)
            cache[threadIdx.x] += cache[threadIdx.x + i];
        __syncthreads();
        i /= 2;
    }
    if(threadIdx.x == 0)
        result[blockIdx.x] = sqrtf(cache[0]);
}

float chatblas_snrm2(int n, float *x)
{
    float *d_x, *results;
    cudaMalloc(&d_x, n*sizeof(float));
    cudaMemcpy(d_x, x, n*sizeof(float), cudaMemcpyHostToDevice);

    cudaMallocManaged(&results, 256*sizeof(float));

    dotproduct<<<256, 256>>>(n, d_x, results);
    cudaDeviceSynchronize();

    float norm = 0;
    for(int i=0;i<256;i++)
        norm += results[i];

    cudaFree(d_x);
    cudaFree(results);

    return norm;
}
