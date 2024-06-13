#include "chatblas_cuda.h"

// CUDA Kernel that processes each element in parallel
__global__ void isamax_kernel(int n, float *x, float *ind)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Shared memory for each block
    __shared__ float data[256];
    __shared__ int ind_data[256];

    data[threadIdx.x] = 0;
    ind_data[threadIdx.x] = -1;

    // Search max absolute value
    for (int i = index; i < n; i += stride)
    {
        float curr_abs_value = fabs(x[i]);
        if (curr_abs_value > data[threadIdx.x])
        {
            data[threadIdx.x] = curr_abs_value;
            ind_data[threadIdx.x] = i;
        }
    }

    __syncthreads();
    
    // Perform a reduction to find max absolute value in the block
    for (int i = blockDim.x / 2; i > 0; i /= 2)
    {
        if (threadIdx.x < i && data[threadIdx.x] < data[threadIdx.x + i])
        {
            data[threadIdx.x] = data[threadIdx.x + i];
            ind_data[threadIdx.x] = ind_data[threadIdx.x + i];
        }
        __syncthreads();
    }

    // Write result for this block to global memory
    if (threadIdx.x == 0)
        ind[blockIdx.x] = ind_data[0];
}

int chatblas_isamax(int n, float *x)
{
    int numBlocks = 256;
    int blockSize = 256;

    float *dev_x = NULL;
    float *dev_Ind = NULL;

    // Allocate memory on GPU
    cudaMalloc((void **)&dev_x, n * sizeof(float));
    cudaMalloc((void **)&dev_Ind, numBlocks * sizeof(float));
    
    // Transfer the vector to the GPU
    cudaMemcpy(dev_x, x, n * sizeof(float), cudaMemcpyHostToDevice);

    isamax_kernel<<<numBlocks, blockSize>>>(n, dev_x, dev_Ind);

    // Copy the index of max absolute value from GPU to CPU
    float *res = (float*)malloc(numBlocks * sizeof(float));
    cudaMemcpy(res, dev_Ind, numBlocks * sizeof(float), cudaMemcpyDeviceToHost);

    int maxInd = 0;
    for(int i = 1; i < numBlocks; ++i)
    {
        if(res[maxInd] < res[i]) maxInd = i;
    }

    cudaFree(dev_x);
    cudaFree(dev_Ind);

    return maxInd;
}