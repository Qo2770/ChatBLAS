#include "chatblas_hip.h"

__global__ void isamax_kernel(int n, float *x, int *ind) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int step_size = gridDim.x * blockDim.x;

    int max_index = -1;
    float max_val = 0.0f;

    for (int i = index; i < n; i += step_size) {
        float val = fabsf(x[i]);
        if (val > max_val) {
            max_val = val;
            max_index = i;
        }
    }

    atomicMax(&ind[0], max_index);
}

int chatblas_isamax(int n, float *x) {
    float *d_x;
    int *d_ind;

    int *h_ind = (int*)malloc(sizeof(int));
    h_ind[0] = -1;

    hipMalloc(&d_x, n * sizeof(float));
    hipMalloc(&d_ind, sizeof(int));
    
    hipMemcpy(d_x, x, n * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_ind, h_ind, sizeof(int), hipMemcpyHostToDevice);

    int block_size = 256;
    int num_blocks = (n + block_size - 1) / block_size;

    isamax_kernel<<<num_blocks, block_size>>>(n, d_x, d_ind);

    hipMemcpy(h_ind, d_ind, sizeof(int), hipMemcpyDeviceToHost);

    int result = h_ind[0];

    free(h_ind);
    hipFree(d_x);
    hipFree(d_ind);

    return result;
}