#include "chatblas_hip.h"

__global__ void isamax_kernel(int n, float *x, int *ind) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float max = 0.0;
    if (i < n) {
        float abs_xi = abs(x[i]);
        if (abs_xi > max) {
            max = abs_xi; 
            ind[0] = i;
        }
    }
}
int chatblas_isamax(int n, float *x) {
    int blockSize = 256; 
    int numBlocks = (n + blockSize - 1) / blockSize;
    float *x_device;
    int *ind_device, ind_host;

    hipMalloc((void **)&x_device, n*sizeof(float));
    hipMalloc((void **)&ind_device, sizeof(int));

    hipMemcpy(x_device, x, n*sizeof(float), hipMemcpyHostToDevice);

    isamax_kernel<<<numBlocks, blockSize>>>(n, x_device, ind_device);
    
    hipMemcpy(&ind_host, ind_device, sizeof(int), hipMemcpyDeviceToHost);

    hipFree(x_device);
    hipFree(ind_device);
    return ind_host;
}