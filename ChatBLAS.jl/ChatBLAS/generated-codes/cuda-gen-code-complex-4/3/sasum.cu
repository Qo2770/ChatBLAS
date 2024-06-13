#include "chatblas_cuda.h"

__global__ void sasum_kernel(int n, float *x, float *sum) {

int i = blockDim.x * blockIdx.x + threadIdx.x;

if (i < n) {
atomicAdd(sum, fabs(x[i]));
}
}

float chatblas_sasum(int n, float *x) {
float *d_x, *d_sum;
float *h_sum = (float*)malloc(sizeof(float));

cudaMalloc((void **)&d_x, sizeof(float)*n);
cudaMalloc((void **)&d_sum, sizeof(float));

cudaMemcpy(d_x, x, sizeof(float)*n, cudaMemcpyHostToDevice);

dim3 DimBlock(256, 1, 1);
dim3 DimGrid((n-1)/256 + 1, 1, 1);

sasum_kernel<<<DimGrid, DimBlock>>>(n, d_x, d_sum);
cudaDeviceSynchronize();

cudaMemcpy(h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);

float sum = *h_sum;
free(h_sum);
cudaFree(d_x); cudaFree(d_sum);

return sum;
}