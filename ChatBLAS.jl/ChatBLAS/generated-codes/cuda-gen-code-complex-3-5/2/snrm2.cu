#include "chatblas_cuda.h"

__global__ void snrm2_kernel(int n, float *x, float *res) {
    // Compute the Euclidean norm of the vector x in parallel
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    float local_sum = 0.0f;
    for (int i = index; i < n; i += stride) {
        local_sum += x[i] * x[i];
    }

    // Perform reduction operation to get the final norm
    __shared__ float shared_sum[256];
    shared_sum[threadIdx.x] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + s];
        }
        __syncthreads();
    }

    // Store the result in the output pointer
    if (threadIdx.x == 0) {
        atomicAdd(res, shared_sum[0]);
    }
}

float chatblas_snrm2(int n, float *x) {
    // Allocate memory for GPU vectors
    float *d_x, *d_res;
    cudaMalloc((void**)&d_x, n * sizeof(float));
    cudaMalloc((void**)&d_res, sizeof(float));

    // Copy input vector from CPU to GPU
    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);

    // Define the size of blocks of threads and the number of blocks
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the kernel
    snrm2_kernel<<<blocksPerGrid, threadsPerBlock>>>(n, d_x, d_res);

    // Copy the result from GPU to CPU
    float result;
    cudaMemcpy(&result, d_res, sizeof(float), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_x);
    cudaFree(d_res);

    // Return the final norm
    return sqrt(result);
}
