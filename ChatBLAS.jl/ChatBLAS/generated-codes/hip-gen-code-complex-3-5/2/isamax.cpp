#include "chatblas_hip.h"
__global__ void isamax_kernel(int n, float *x, int *ind) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(i < n){
        if(fabs(x[i]) > fabs(x[ind[0]])){
            ind[0] = i;
        }
    }
}

int chatblas_isamax(int n, float *x) {
    float *d_x;
    int *d_ind;
    int *ind = (int*)malloc(sizeof(int));
    
    hipMalloc(&d_x, n * sizeof(float));
    hipMalloc(&d_ind, sizeof(int));
    
    hipMemcpy(d_x, x, n * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_ind, ind, sizeof(int), hipMemcpyHostToDevice);
    
    int block_size = 256;
    int num_blocks = (n + block_size - 1) / block_size;
    
    isamax_kernel<<<num_blocks, block_size>>>(n, d_x, d_ind);
    
    hipMemcpy(ind, d_ind, sizeof(int), hipMemcpyDeviceToHost);
    
    hipFree(d_x);
    hipFree(d_ind);
    
    int result = ind[0];
    free(ind);
    
    return result;
}