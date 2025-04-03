#include "chatblas_cuda.h"

__global__ void snrm2_kernel(int n, float *x, float *res) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    __shared__ float sum[256]; // PEDRO: this line was changed

    // Initializing local shared array
    sum[threadIdx.x] = 0.0f;
    if(idx < n) {
        sum[threadIdx.x] = x[idx] * x[idx];
    }

    __syncthreads();
    // Reduce
    for (int s=blockDim.x/2; s>0; s>>=1) {
        if (threadIdx.x < s) {
            sum[threadIdx.x] += sum[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(res, sum[0]);
    }
}

float chatblas_snrm2(int n, float *x) {
    const int threads = 256;
    const int blocks = min((n + threads - 1) / threads, 1024);
    float *device_x, *device_res, result;

    cudaMalloc((void **)&device_x, n * sizeof(float));
    cudaMalloc((void **)&device_res, blocks * sizeof(float));

    cudaMemcpy(device_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(device_res, 0, blocks * sizeof(float));

    snrm2_kernel<<<blocks, threads>>>(n, device_x, device_res);

    float *cpu_res = (float *)malloc(blocks * sizeof(float));
    cudaMemcpy(cpu_res, device_res, blocks * sizeof(float), cudaMemcpyDeviceToHost);

    result = sqrt(cpu_res[0]);

    cudaFree(device_x);
    cudaFree(device_res);
    free(cpu_res);

    return result;
}
