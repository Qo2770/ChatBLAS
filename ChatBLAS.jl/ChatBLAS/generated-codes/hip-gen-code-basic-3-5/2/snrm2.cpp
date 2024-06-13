#include "chatblas_hip.h"

__global__ void snrm2_kernel( int n, float *x, float *res){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < n){
        atomicAdd(res, x[tid]*x[tid]);
    }
}

float chatblas_snrm2(int n, float *x) {
    float *d_x, *d_res, h_res;
    float *h_x = (float*)malloc(n * sizeof(float));
    for(int i=0; i<n; i++){
        h_x[i] = x[i];
    }
    hipMalloc(&d_x, n * sizeof(float));
    hipMalloc(&d_res, sizeof(float));
    hipMemcpy(d_x, h_x, n * sizeof(float), hipMemcpyHostToDevice);

    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    
    snrm2_kernel<<<grid_size, block_size>>>(n, d_x, d_res);
    
    hipMemcpy(&h_res, d_res, sizeof(float), hipMemcpyDeviceToHost);

    free(h_x);
    hipFree(d_x);
    hipFree(d_res);
    
    return sqrt(h_res);
}