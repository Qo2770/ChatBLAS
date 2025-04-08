#include "chatblas_cuda.h"

__global__ void vectorSquare(float *x, float *result, int n) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n) {
        result[index] = x[index] * x[index];
    }
}

float chatblas_snrm2(int n, float *x) {
    float *d_x, *d_result, *h_result;
    float norm = 0.0f;

    // Allocate memory on the device
    cudaMalloc((void**)&d_x, n * sizeof(float));
    cudaMalloc((void**)&d_result, n * sizeof(float));
    h_result = (float*)malloc(n * sizeof(float));

    // Copy the data to the device
    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);

    // Kernel launch parameters
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    // Launch the kernel to compute the square of each element
    vectorSquare<<<gridSize, blockSize>>>(d_x, d_result, n);

    // Copy the result back to host
    cudaMemcpy(h_result, d_result, n * sizeof(float), cudaMemcpyDeviceToHost);

    // Sum the squared elements on the host
    for (int i = 0; i < n; i++) {
        norm += h_result[i];
    }

    // Take the square root of the sum
    norm = sqrtf(norm);

    // Free device memory
    cudaFree(d_x);
    cudaFree(d_result);
    // Free host memory
    free(h_result);

    return norm;
}
