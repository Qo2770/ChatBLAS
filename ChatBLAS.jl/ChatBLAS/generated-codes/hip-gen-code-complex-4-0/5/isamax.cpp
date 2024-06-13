#include "chatblas_hip.h"

__global__ void isamax_kernel(int n, float *x, int *ind) {

    int i = threadIdx.x + blockIdx.x * blockDim.x;

    __shared__ float aux[256];
    __shared__ int   index[256];

    if(i < n) {
        aux[threadIdx.x]    = abs(x[i]);
        index[threadIdx.x]  = i;
    }
    __syncthreads();
    
    for(int s = blockDim.x / 2; s > 0; s /= 2) {
        if(threadIdx.x < s && i+s < n) {
            if(aux[threadIdx.x+s] > aux[threadIdx.x]) {
                aux[threadIdx.x]   = aux[threadIdx.x+s];
                index[threadIdx.x] = index[threadIdx.x+s];
            }
        }
        __syncthreads();
    }
    
    if(threadIdx.x == 0) ind[blockIdx.x] = index[0]; 
}

int chatblas_isamax(int n, float *x) {
    float *d_x;
    int *d_ind;

    hipMalloc((void **)&d_x, n*sizeof(float));
    hipMalloc((void **)&d_ind, n*sizeof(int));

    hipMemcpy(d_x, x, n*sizeof(float), hipMemcpyHostToDevice);

    int numBlocks = (n + 255) / 256;
    int numThreads = 256;
    isamax_kernel<<< numBlocks, numThreads >>>(n, d_x, d_ind);

    int* ind = (int *)malloc(n * sizeof(int));
    hipMemcpy(ind, d_ind, n * sizeof(int), hipMemcpyDeviceToHost);

    int maxindex = 0;
    for(int i = 1; i < num_blocks; ++i)
      if(x[ind[i]] > x[ind[maxindex]])
        maxindex = i;

    hipFree(d_x);
    hipFree(d_ind);
  
    int result = ind[maxindex];

    free(ind);

    return result;
}