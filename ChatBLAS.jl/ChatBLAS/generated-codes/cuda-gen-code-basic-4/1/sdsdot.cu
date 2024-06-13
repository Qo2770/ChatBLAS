#include "chatblas_cuda.h"

// CUDA Kernel that calculates dot product
__global__ void dotProductKernel(float *x, float *y, float *dot, int n){
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for(int i=index; i<n; i+=stride)
        dot[index] += ((double)x[i]) * ((double)y[i]);
}

// Sum reduction function
__device__ void reduction(float *input, int n){
    int threadId = threadIdx.x;
    int blockSize = blockDim.x;
	
	int i = blockSize / 2;
	while(i != 0){
		if(threadId < i){
			input[threadId] += input[threadId + i];
		}
		__syncthreads();
		i /= 2;
	}
}

float chatblas_sdsdot(int n, float b, float *x, float *y){

    int size = n*sizeof(float);
    float *dot;
    float *d_x, *d_y, *d_dot;

    dot = (float*)malloc(size);

    cudaMalloc(&d_x, size);
    cudaMalloc(&d_y, size);
    cudaMalloc(&d_dot, size);

    cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, size, cudaMemcpyHostToDevice);

    int blockSize = 1024;
    int numBlocks = (n + blockSize - 1) / blockSize;
    dotProductKernel<<<numBlocks, blockSize>>>(d_x, d_y, d_dot, n);
	
    reduction<<<1, blockSize>>>(d_dot, n);
	
    cudaMemcpy(dot, d_dot, size, cudaMemcpyDeviceToHost);
	
    float result = dot[0] + b;

    free(dot);
    cudaFree(d_x); cudaFree(d_y); cudaFree(d_dot);

    return result;
}
