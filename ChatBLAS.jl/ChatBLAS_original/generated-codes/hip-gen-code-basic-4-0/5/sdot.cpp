#include "chatblas_hip.h"

__global__ void sdot_kernel(int n, float *x, float *y, float *res) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n)
        res[idx] = x[idx] * y[idx];
}

float chatblas_sdot( int n, float *x, float *y) {
    float *x_d, *y_d, *res_d, *res;
    float sum = 0.0f;
    size_t size = n * sizeof(float);

    res = (float*)malloc(size);

    hipMalloc(&x_d, size);
    hipMalloc(&y_d, size);
    hipMalloc(&res_d, size);

    hipMemcpy(x_d, x, size, hipMemcpyHostToDevice);
    hipMemcpy(y_d, y, size, hipMemcpyHostToDevice);

    // Assuming Block size of 256
    sdot_kernel<<<(n + 255) / 256, 256>>>(n, x_d, y_d, res_d);

    hipMemcpy(res, res_d, size, hipMemcpyDeviceToHost);

    for(int i = 0; i < n; i++) {
        sum += res[i];
    }

    free(res);
    hipFree(x_d);
    hipFree(y_d);
    hipFree(res_d);

    return sum;
}
