#include "chatblas_cuda.h"

// CUDA Kernel function to compute the dot product of two vectors
__global__ void chatblas_sdsdot_kernel(int n, float b, float *x, float *y, float *dotProd)
{
    // Calculating the unique ID for each thread
    unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;

    // Thread bounds check
    if(i < n) 
    {
        double x_elem = (double)x[i];
        double y_elem = (double)y[i];
        double prod = x_elem * y_elem;

        atomicAdd(dotProd, (float)prod);
    }
}

float chatblas_sdsdot(int n, float b, float *x, float *y)
{
    float *dev_x, *dev_y, *dev_dotProd;
    float dotProd = 0.0;

    cudaMalloc((void **)&dev_x, n * sizeof(float));
    cudaMalloc((void **)&dev_y, n * sizeof(float));
    cudaMalloc((void **)&dev_dotProd, sizeof(float));

    cudaMemcpy(dev_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_y, y, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_dotProd, &dotProd, sizeof(float), cudaMemcpyHostToDevice);

    // Number of threads per block and number of blocks per grid
    int threadsPerBlock = 256;
    int blocksPerGrid =(n + threadsPerBlock - 1) / threadsPerBlock;

    chatblas_sdsdot_kernel<<<blocksPerGrid, threadsPerBlock>>>(n, b, dev_x, dev_y, dev_dotProd);

    // Copy result back to host
    cudaMemcpy(&dotProd, dev_dotProd, sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(dev_x);
    cudaFree(dev_y);
    cudaFree(dev_dotProd);

    // Adding scalar b to the dot product
    dotProd += b;

    return dotProd;
}
