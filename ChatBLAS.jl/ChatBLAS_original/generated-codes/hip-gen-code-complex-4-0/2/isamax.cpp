#include "chatblas_hip.h"

__global__ void isamax_kernel(int n, float *x, int *ind) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    extern __shared__ float cache[];
    float temp = 0.0;
    int   temp_ind = 0;

    while(idx < n) {
        if(abs(x[idx]) > temp) {
            temp = abs(x[idx]);
            temp_ind = idx;
        }
        idx += blockDim.x * gridDim.x;
    }
    cache[threadIdx.x] = temp;
    __syncthreads();

    int i = blockDim.x / 2;
    while(i != 0) {
        if(threadIdx.x < i) {
            if(cache[threadIdx.x] < cache[threadIdx.x+i]) {
                cache[threadIdx.x] = cache[threadIdx.x+i];
                temp_ind = idx + i;
            }
        }
        __syncthreads();
        i /= 2;
    }

    if(threadIdx.x == 0) {
        ind[blockIdx.x] = temp_ind;
    }
}

int chatblas_isamax(int n, float *x) {
    float* d_x;
    int* d_ind;
    int ind;

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    hipMalloc((void**)&d_x, n * sizeof(float));
    hipMemcpy(d_x, x, n * sizeof(float), hipMemcpyHostToDevice);

    hipMalloc((void**)&d_ind, blocksPerGrid * sizeof(int));

    isamax_kernel<<<blocksPerGrid, threadsPerBlock>>>(n, d_x, d_ind);

    std::vector<int> h_ind(blocksPerGrid);
    hipMemcpy(h_ind.data(), d_ind, blocksPerGrid * sizeof(int), hipMemcpyDeviceToHost);

    ind = std::distance(h_ind.begin(), max_element(h_ind.begin(), h_ind.end()));

    hipFree(d_x);
    hipFree(d_ind);

    return ind;
}