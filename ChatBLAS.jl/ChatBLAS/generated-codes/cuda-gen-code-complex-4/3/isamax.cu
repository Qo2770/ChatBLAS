#include "chatblas_cuda.h"

__global__ void isamax_kernel(int n, float *x, float *ind) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    __shared__ float temp[BLOCKSIZE];
    __shared__ int   indexes[BLOCKSIZE];
    temp[threadIdx.x] = 0;

    while(idx < n) {
        float current = fabsf(x[idx]);
        if(current > temp[threadIdx.x]) {
            temp[threadIdx.x] = current;
            indexes[threadIdx.x] = idx;
        }
        idx += gridDim.x * blockDim.x;
    }

    __syncthreads();
    for(int s = blockDim.x / 2; s > 0; s >>= 1) {
        if(threadIdx.x < s) {
            if(temp[threadIdx.x] < temp[threadIdx.x + s]) {
                temp[threadIdx.x] = temp[threadIdx.x + s];
                indexes[threadIdx.x] = indexes[threadIdx.x + s];
            }
        }
        __syncthreads();
    }

    if(threadIdx.x == 0) ind[blockIdx.x] = indexes[0];
}

int chatblas_isamax(int n, float *x) {
    float *d_x, *d_ind;
    int blockSize = BLOCKSIZE;
    int numBlocks = (n + BLOCKSIZE - 1) / BLOCKSIZE;
    int ind;
 
    // Allocate device memory
    cudaMalloc(&d_x, n * sizeof(float));
    cudaMalloc(&d_ind, numBlocks * sizeof(float));

    // Copy vectors x to the device memory
    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);

    // Launch the kernel
    isamax_kernel<<<numBlocks, blockSize>>>(n, d_x, d_ind);

    // Allocate host memory
    float *h_ind = (float *)malloc(numBlocks * sizeof(float));
 
    // Copy the content from the device memory to vectors h_ind
    cudaMemcpy(h_ind, d_ind, numBlocks * sizeof(float), cudaMemcpyDeviceToHost);

    // Find the maximum index in the host memory
    ind = h_ind[0];
    for(int i = 1; i < numBlocks; i++) {
        if (x[(int)ind] < x[(int)h_ind[i]]) ind = (int)h_ind[i];
    }

    // Free device and host memory
    cudaFree(d_x);
    cudaFree(d_ind);
    free(h_ind);

    return ind;
}
