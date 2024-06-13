#include "chatblas_cuda.h"

__global__ void isamax_kernel(int n, float *x, float *ind) {

    int i = blockDim.x * blockIdx.x + threadIdx.x; 

    if (i < n) {
      if (abs(x[i]) > abs(x[(int)(*ind)])) {
        *ind = (float)i;
      }
    }
}

int chatblas_isamax(int n, float *x) {
 
    float *x_device, *ind_device;
    float ind_host = 0; 

    cudaMalloc((void**)&x_device, n * sizeof(float));
    cudaMalloc((void**)&ind_device, sizeof(float));

    cudaMemcpy(x_device, x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(ind_device, &ind_host, sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    isamax_kernel<<<gridSize, blockSize>>>(n, x_device, ind_device);
    
    cudaMemcpy(&ind_host, ind_device, sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(x_device);
    cudaFree(ind_device);
      
    return (int) ind_host;
}