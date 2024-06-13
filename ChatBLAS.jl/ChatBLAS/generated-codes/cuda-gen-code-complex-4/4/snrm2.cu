#include "chatblas_cuda.h"

__global__ void snrm2_kernel(int n, float *x, float *res) {
    extern __shared__ float shared_data[];
    const unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + tid;
    shared_data[tid] = (i < n) ? x[i] * x[i] : 0.0f;

    __syncthreads();

    for(unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if(tid < stride)
            shared_data[tid] += shared_data[tid + stride];

        __syncthreads();
    }

    if(tid == 0)
        atomicAdd(res, shared_data[0]);
}

float chatblas_snrm2(int n, float *x) {
    float *d_x, *d_res;
    float result=0.0f;
    size_t size = n * sizeof(float);

    cudaMalloc((void **)&d_x, size);
    cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice);

    cudaMalloc((void **)&d_res, sizeof(float));
    cudaMemcpy(d_res, &result, sizeof(float), cudaMemcpyHostToDevice);

    dim3 dimBlock(256, 1, 1);
    dim3 dimGrid((n+dimBlock.x-1)/dimBlock.x , 1, 1);

    snrm2_kernel<<<dimGrid, dimBlock, 256*sizeof(float)>>>(n, d_x, d_res);

    cudaMemcpy(&result, d_res, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_res);

    return sqrtf(result);
}