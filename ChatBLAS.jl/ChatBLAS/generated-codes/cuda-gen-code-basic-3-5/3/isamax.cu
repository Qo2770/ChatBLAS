#include "chatblas_cuda.h"

__global__ void findMaxAbsValue(float *x, int n, int *maxIndex) {
    float maxAbsValue = 0.0;
    int index = 0;

    // Compute the index of the element with the maximum absolute value
    for(int i = threadIdx.x; i < n; i += blockDim.x) {
        float absValue = fabsf(x[i]);
        if(absValue > maxAbsValue) {
            maxAbsValue = absValue;
            index = i;
        }
    }

    // Perform block-level reduction to find the global maximum absolute value
    for(int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if(threadIdx.x < stride) {
            float absValue = fabsf(x[index + stride]);
            if(absValue > maxAbsValue) {
                maxAbsValue = absValue;
                index += stride;
            }
        }
        __syncthreads();
    }

    // Store the global maximum index in shared memory
    if(threadIdx.x == 0) {
        *maxIndex = index;
    }
}

int chatblas_isamax(int n, float *x) {
    int *d_maxIndex, maxIndex;
    float *d_x;

    // Allocate device memory for the input vector
    cudaMalloc((void**)&d_x, n * sizeof(float));

    // Allocate device memory for the maximum index
    cudaMalloc((void**)&d_maxIndex, sizeof(int));

    // Copy the input vector to device memory
    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);

    // Launch the kernel to find the maximum absolute value
    findMaxAbsValue<<<1, 256>>>(d_x, n, d_maxIndex);

    // Copy the maximum index from device to host
    cudaMemcpy(&maxIndex, d_maxIndex, sizeof(int), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_x);
    cudaFree(d_maxIndex);

    // Return the index of the element with the maximum absolute value
    return maxIndex;
}