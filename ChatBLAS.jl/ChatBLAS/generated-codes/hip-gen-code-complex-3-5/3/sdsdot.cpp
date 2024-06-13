#include "chatblas_hip.h"

__global__ void sdsdot_kernel( int n, float b, float *x, float *y, float *res ) {
    __shared__ double shared_data[256]; // Assuming a block size of 256 threads
    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    double acc = 0.0;
    while (index < n) {
        acc += (double)((float)x[index]) * (double)((float)y[index]);
        index += blockDim.x * gridDim.x;
    }
    shared_data[tid] = acc;
    
    __syncthreads();
    
    // Reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd( res, shared_data[0] + (double)b );
    }
}

float chatblas_sdsdot( int n, float b, float *x, float *y) {
    float *d_x, *d_y, *d_res;
    float h_res;
    
    hipMalloc( &d_x, n * sizeof(float) );
    hipMalloc( &d_y, n * sizeof(float) );
    hipMalloc( &d_res, sizeof(float) );
    
    hipMemcpy( d_x, x, n * sizeof(float), hipMemcpyHostToDevice );
    hipMemcpy( d_y, y, n * sizeof(float), hipMemcpyHostToDevice );
    
    sdsdot_kernel<<<numBlocks, blockSize>>>( n, b, d_x, d_y, d_res );
    
    hipMemcpy( &h_res, d_res, sizeof(float), hipMemcpyDeviceToHost );
    
    hipFree( d_x );
    hipFree( d_y );
    hipFree( d_res );
    
    return h_res;
}
