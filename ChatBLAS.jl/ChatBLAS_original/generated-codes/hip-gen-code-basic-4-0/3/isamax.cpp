#include "chatblas_hip.h"

__global__ void isamax_kernel(int n, float *x, int *ind) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    __shared__ float max_val;
    __shared__ int max_ind;
    if(i == 0) {
        max_val = fabs(x[0]);
        max_ind = 0;
    }
    __syncthreads();
    if(i < n) {
        float val = fabs(x[i]);
        if(val > max_val) {
            max_val = val;
            max_ind = i;
        }
    }
    __syncthreads();
    if(i == 0) {
        *ind = max_ind;
    }
}

int chatblas_isamax(int n, float *x) {
    float *dev_x;
    int *dev_ind, ind;
    hipMalloc((void**) &dev_x, n * sizeof(float));
    hipMalloc((void**) &dev_ind, sizeof(int));

    hipMemcpy(dev_x, x, n * sizeof(float), hipMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid =(n + threadsPerBlock - 1) / threadsPerBlock;
    isamax_kernel<<<blocksPerGrid, threadsPerBlock>>>(n, dev_x, dev_ind);

    hipMemcpy(&ind, dev_ind, sizeof(int), hipMemcpyDeviceToHost);

    hipFree(dev_x);
    hipFree(dev_ind);

    return ind;
}
