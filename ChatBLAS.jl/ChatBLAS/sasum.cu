#include "chatblas_cuda.h" 
__global__ void sasum_kernel(int n, float *x, float *sum) { 
	__shared__ float cache[256]; 
	int i = blockIdx.x * blockDim.x + threadIdx.x; 
	int stride = blockDim.x * gridDim.x; 
	float local_sum = 0.0f; 
	for (int j = i; j < n; j += stride) { 
		local_sum += fabsf(x[j]); 
	} 
	cache[threadIdx.x] = local_sum; 
	__syncthreads(); 
	int k = blockDim.x/2; 
	while (k != 0) { 
		if (threadIdx.x < k) { 
			cache[threadIdx.x] += cache[threadIdx.x + k]; 
		} 
		__syncthreads(); 
		k /= 2; 
	} 
	if (threadIdx.x == 0) { 
		atomicAdd(sum, cache[0]); 
	} 
} 

float chatblas_sasum(int n, float *x) { 
	float *d_x, *d_sum; 
	float sum = 0.0f; 
	cudaMalloc(&d_x, n * sizeof(float)); 
	cudaMalloc(&d_sum, sizeof(float)); 
	cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice); 
	cudaMemcpy(d_sum, &sum, sizeof(float), cudaMemcpyHostToDevice); 
	int blockSize = 256; 
	int numBlocks = (n + blockSize - 1) / blockSize; 
	sasum_kernel<<<numBlocks, blockSize>>>(n, d_x, d_sum); 
	cudaMemcpy(&sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost); 
	cudaFree(d_x); 
	cudaFree(d_sum); 
	return sum; 
}
