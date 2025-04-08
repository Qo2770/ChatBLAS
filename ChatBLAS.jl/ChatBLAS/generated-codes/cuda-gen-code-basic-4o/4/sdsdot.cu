#include "chatblas_cuda.h"
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void dotProductKernel(int n, const float *x, const float *y, double *partial_sum) {
    extern __shared__ double sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Each thread computes one element of the dot product.
    double temp = 0.0;
    if (i < n) {
        temp = (double)x[i] * (double)y[i];
    }
    
    // Store the result into shared memory.
    sdata[tid] = temp;
    __syncthreads();

    // Reduce within block.
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write block's results to global memory to accumulate.
    if (tid == 0) {
        partial_sum[blockIdx.x] = sdata[0];
    }
}

float chatblas_sdsdot(int n, float b, float *x, float *y) {
    // Allocate device memory
    float *d_x, *d_y;
    double *d_partial_sum;
    cudaMalloc((void**)&d_x, n * sizeof(float));
    cudaMalloc((void**)&d_y, n * sizeof(float));

    int blocks = (n + 255) / 256;
    cudaMalloc((void**)&d_partial_sum, blocks * sizeof(double));

    // Copy host vectors to device
    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, n * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dotProductKernel<<<blocks, 256, 256 * sizeof(double)>>>(n, d_x, d_y, d_partial_sum);

    // Allocate array for partial sums on host
    double h_partial_sum[blocks];

    // Copy partial sums back to host
    cudaMemcpy(h_partial_sum, d_partial_sum, blocks * sizeof(double), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_partial_sum);

    // Sum up partial sums
    double dot = 0.0;
    for (int i = 0; i < blocks; i++) {
        dot += h_partial_sum[i];
    }

    // Add scalar b
    dot += (double)b;

    return (float)dot;
}

int main() {
    // Example usage; adjust size accordingly
    int n = 1000;
    float *x, *y;
    x = (float *)malloc(n * sizeof(float));
    y = (float *)malloc(n * sizeof(float));

    for (int i = 0; i < n; ++i) {
        x[i] = 1.0f;
        y[i] = 1.0f;
    }

    float b = 1.5f;
    float result = chatblas_sdsdot(n, b, x, y);
    printf("Result: %f\n", result);

    free(x);
    free(y);

    return 0;
}
