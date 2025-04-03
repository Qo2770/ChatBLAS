#include <cub/cub.cuh>
#include "chatblas_cuda.h"

__global__ void dotProductKernel(int n, float *x, float *y, float *result) {
    extern __shared__ float temp[];

	int index = threadIdx.x;
	int stride = blockDim.x;

	temp[index] = 0;

	if (index < n) {
		for (int i = index; i < n; i += stride)
			temp[index] += x[i] * y[i];
	}

	__syncthreads();

	// Use CUB's DeviceReduce to get the final dot product result
	cub::DeviceReduce::Sum(temp, result, n);
}

float chatblas_sdot(int n, float *x, float *y) {
    float *d_x, *d_y, *d_result;
    float result;

    // allocate memory on the device
    cudaMalloc(&d_x, n * sizeof(float));
    cudaMalloc(&d_y, n * sizeof(float));
    cudaMalloc(&d_result, sizeof(float));

    // copy inputs to device
    cudaMemcpy(d_x, x, n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, n*sizeof(float), cudaMemcpyHostToDevice);

    // launch kernel
    dotProductKernel<<<1, 256, 256*sizeof(float)>>>(n, d_x, d_y, d_result);

    // copy result back to host
    cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost);

    // cleanup
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_result);

    return result;
}
