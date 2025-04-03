#include "chatblas_cuda.h"

// definition of the kernel function
__global__ void sdsdot_kernel( int n, float b, float *x, float *y, float *res ) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < n) {
        double temp = (double)x[index] * (double)y[index];
        atomicAdd(res, temp);
    }
}

// definition of the function that calls the kernel 
float chatblas_sdsdot( int n, float b, float *x, float *y) {
    float res, *gpu_res;
    float *gpu_x, *gpu_y;

    int threads_per_block = 256;
    int blocks_per_grid = (n + threads_per_block - 1) / threads_per_block;

    // allocate memory on the GPU for the vectors and the result
    cudaMalloc((void**)&gpu_x, n * sizeof(float));
    cudaMalloc((void**)&gpu_y, n * sizeof(float));
    cudaMalloc((void**)&gpu_res, sizeof(float));

    // copy from CPU to GPU memory
    cudaMemcpy(gpu_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_y, y, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(gpu_res, 0, sizeof(float));

    // launch the kernel
    sdsdot_kernel<<<blocks_per_grid, threads_per_block>>>(n, b, gpu_x, gpu_y, gpu_res);

    // wait for GPU to finish before accessing on CPU
    cudaDeviceSynchronize();

    // copy result from GPU to CPU
    cudaMemcpy(&res, gpu_res, sizeof(float), cudaMemcpyDeviceToHost);
    
    // cleanup
    cudaFree(gpu_x);
    cudaFree(gpu_y);
    cudaFree(gpu_res);

    return res + b;
}