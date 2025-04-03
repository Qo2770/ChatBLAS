#include "chatblas_cuda.h"

__global__ void sasum_kernel(int n, float *x, float *sum) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    extern __shared__ float temp[];

    temp[index] = (index < n) ? fabs(x[index]) : 0;
    __syncthreads();

    if (index == 0) {
        float totalSum = 0;
        for (int i = 0; i < blockDim.x; i++) {
            totalSum += temp[i];
        }
        *sum = totalSum;
    }
}

float chatblas_sasum(int n, float *x) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    float *d_x, *d_sum;
  
    // Allocate memory for arrays on GPU
    cudaMalloc((void **)&d_x, n * sizeof(float));
    cudaMalloc((void **)&d_sum, sizeof(float));

    // Copy array from host to device
    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);

    // Execute kernel
    sasum_kernel<<<blocksPerGrid, threadsPerBlock>>>(n, d_x, d_sum);

    // Allocate memory on host to store result
    float host_sum;
    cudaMemcpy(&host_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_x);
    cudaFree(d_sum);

    return host_sum;
}
