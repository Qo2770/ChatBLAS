#include "chatblas_hip.h"

__global__ void sscal_kernel( int n, float a , float *x ) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if ( i < n ) {
        x[i] = a * x[i];
    }
}

void chatblas_sscal( int n, float a, float *x){
    int size = n * sizeof(float);
    float *x_gpu;
    
    // Allocate GPU memory
    HIP_CHECK(hipMalloc((void**) &x_gpu, size));
    
    // Copy the vector x from CPU to GPU
    HIP_CHECK(hipMemcpy(x_gpu, x, size, hipMemcpyHostToDevice));

    // Set grid and block dimensions
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    
    // Call the kernel
    hipLaunchKernelGGL(
        sscal_kernel,
        dim3(gridSize),
        dim3(blockSize),
        0, 0,
        n, a, x_gpu
    );
    
    // Wait for GPU to finish before returning control to CPU
    HIP_CHECK(hipDeviceSynchronize());

    // Copy the result vector back to the CPU from the GPU
    HIP_CHECK(hipMemcpy(x, x_gpu, size, hipMemcpyDeviceToHost));
    
    // Free the GPU memory
    HIP_CHECK(hipFree(x_gpu));
}
