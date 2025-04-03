#include "chatblas_hip.h"
__global__ void isamax_kernel(int n, float *x, int *ind) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int maxIdx = 0;
    float maxVal = 0.0f;
    
    for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
        float val = fabs(x[i]);
        if (val > maxVal) {
            maxVal = val;
            maxIdx = i;
        }
    }
    
    atomicMaxf(maxVal, maxVal);
    
    if (tid == 0) {
        *ind = maxIdx;
    }
}

int chatblas_isamax(int n, float *x) {
    float *d_x; 
    int *d_ind;
    int ind; 

    hipMalloc(&d_x, n * sizeof(float));
    hipMemcpy(d_x, x, n * sizeof(float), hipMemcpyHostToDevice);
    
    hipMalloc(&d_ind, sizeof(int));

    int numBlocks = 256;
    int blockSize = 256;

    isamax_kernel<<<numBlocks, blockSize>>>(n, d_x, d_ind);

    hipMemcpy(&ind, d_ind, sizeof(int), hipMemcpyDeviceToHost);

    hipFree(d_x);
    hipFree(d_ind);

    return ind;
}
