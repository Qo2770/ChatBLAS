#include "chatblas_hip.h"

__global__ void isamax_kernel(int n, float *x, int *ind) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    __shared__ float maxval[256];
    __shared__ int maxind[256];
  
    if(i < n)
    {
        maxval[threadIdx.x] = fabsf(x[i]);
        maxind[threadIdx.x] = i;
    }
    else
    {
        maxval[threadIdx.x] = 0.0;
        maxind[threadIdx.x] = -1;
    }
  
    __syncthreads();

    int threadnum = blockDim.x;
    while(threadnum > 1)
    {
        int halfpoint = (threadnum >> 1);
        if(threadIdx.x < halfpoint)
        {
            if(maxval[threadIdx.x + halfpoint] > maxval[threadIdx.x])
            {
                maxval[threadIdx.x] = maxval[threadIdx.x + halfpoint];
                maxind[threadIdx.x] = maxind[threadIdx.x + halfpoint];
            }
        }
        __syncthreads();
        threadnum = halfpoint;
    }

    if(threadIdx.x == 0)
    {
        ind[blockIdx.x] = maxind[0];
    }
}

int chatblas_isamax(int n, float *x) {
    int num_blocks = (n + 255) / 256;
    int *d_ind, *h_ind;
    float *d_x;
    h_ind = (int*)malloc(num_blocks*sizeof(int));

    hipMalloc((void**)&d_x, n*sizeof(float));
    hipMalloc((void**)&d_ind, num_blocks*sizeof(int));

    hipMemcpy(d_x, x, n*sizeof(float), hipMemcpyHostToDevice);

    isamax_kernel<<<num_blocks, 256>>>(n, d_x, d_ind);

    hipMemcpy(h_ind, d_ind, num_blocks*sizeof(int), hipMemcpyDeviceToHost);

    hipFree(d_x);
    hipFree(d_ind);

    int i, maxind = h_ind[0];
    for (i = 1; i < num_blocks; i++)
    {
        if(x[h_ind[i]] > x[maxind])
        {
            maxind = h_ind[i];
        }
    }

    free(h_ind);

    return maxind;
}
