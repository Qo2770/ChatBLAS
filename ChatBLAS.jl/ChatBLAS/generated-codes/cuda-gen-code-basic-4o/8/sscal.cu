#include "chatblas_cuda.h"
#include <cuda_runtime.h>
#include <stdio.h>

// CUDA Kernel to scale a vector
__global__ void scale_vector_kernel(float *d_x, int n, float a) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        d_x[index] *= a;
    }
}

// Function to perform the scaling on the host
void chatblas_sscal(int n, float a, float *x) {
    float *d_x;
    size_t size = n * sizeof(float);

    // Allocate memory on the GPU
    cudaMalloc((void **)&d_x, size);

    // Copy data from host to device
    cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice);

    // Define the number of threads and blocks
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    // Launch the kernel
    scale_vector_kernel<<<numBlocks, blockSize>>>(d_x, n, a);

    // Copy result back to host
    cudaMemcpy(x, d_x, size, cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_x);
}

// Example usage
int main() {
    int n = 1000;
    float a = 5.0f;
    float *x = (float *)malloc(n * sizeof(float));

    // Initialize the vector
    for (int i = 0; i < n; i++) {
        x[i] = 1.0f;  // Example initialization
    }

    // Call the chatblas_sscal function
    chatblas_sscal(n, a, x);

    // Output the result for verification
    for (int i = 0; i < n; i++) {
        printf("%f ", x[i]);
    }

    // Free host memory
    free(x);

    return 0;
}
