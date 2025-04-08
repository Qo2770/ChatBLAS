#include "chatblas_cuda.h"

__global__ void find_max_abs_index(int n, float *x, int *result) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < n) {
        sdata[tid] = fabsf(x[index]);
    } else {
        sdata[tid] = -1.0f;  // Use a negative value for non-existing elements
    }
    __syncthreads();

    // Perform reduction to find the index of the maximum absolute value
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && index + s < n) {
            if (sdata[tid] < sdata[tid + s]) {
                sdata[tid] = sdata[tid + s];
            }
        }
        __syncthreads();
    }

    // Write the result for this block to the output
    if (tid == 0) {
        result[blockIdx.x] = blockIdx.x * blockDim.x + tid;
    }
}

int chatblas_isamax(int n, float *x) {
    if (n <= 0) return -1;

    int *d_result, *h_result;
    float *d_x;
    int blocks = (n + 255) / 256;
    size_t size_int = blocks * sizeof(int);

    cudaMalloc((void**)&d_x, n * sizeof(float));
    cudaMalloc((void**)&d_result, size_int);
    h_result = (int*)malloc(size_int);

    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);

    find_max_abs_index<<<blocks, 256, 256 * sizeof(float)>>>(n, d_x, d_result);

    cudaMemcpy(h_result, d_result, size_int, cudaMemcpyDeviceToHost);

    // Find the final index on CPU
    int max_index = h_result[0];
    float max_value = fabsf(x[max_index]);
    for (int i = 1; i < blocks; i++) {
        int index = h_result[i];
        if (index < n && fabsf(x[index]) > max_value) {
            max_index = index;
            max_value = fabsf(x[max_index]);
        }
    }

    cudaFree(d_x);
    cudaFree(d_result);
    free(h_result);

    return max_index;
}
