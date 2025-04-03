#include "chatblas_cuda.h"

__global__ void scaleVector(int n, float a, float *x) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        x[idx] *= a;
    }
}

void chatblas_sscal(int n, float a, float *x) {
    float *dev_x;
    if (cudaMalloc((void **)&dev_x, n * sizeof(float)) != cudaSuccess) {
        printf("Failed to allocate device memory for x.\n");
        return;
    }

    if (cudaMemcpy(dev_x, x, n * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
        printf("Failed to copy x to device memory.\n");
        cudaFree(dev_x);
        return;
    }

    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    scaleVector<<<gridSize, blockSize>>>(n, a, dev_x);

    if (cudaMemcpy(x, dev_x, n * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
        printf("Failed to copy scaled x back to host memory.\n");
    }

    cudaFree(dev_x);
}
