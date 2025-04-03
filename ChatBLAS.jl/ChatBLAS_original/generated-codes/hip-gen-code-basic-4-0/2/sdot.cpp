#include "chatblas_hip.h"

__global__ void sdot_kernel(int n, float *d_x, float *d_y, float *d_res) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    if (index<n) {
        atomicAdd(d_res, d_x[index]*d_y[index]);
    }
}

float chatblas_sdot( int n, float *host_x, float *host_y) {
    float *device_x, *device_y, host_result = 0.0f;
    float *device_result;

    hipMalloc( &device_x, n*sizeof(float));
    hipMemcpy( device_x, host_x, n*sizeof(float), hipMemcpyHostToDevice);

    hipMalloc( &device_y, n*sizeof(float));
    hipMemcpy( device_y, host_y, n*sizeof(float), hipMemcpyHostToDevice);

    hipMalloc( &device_result, sizeof(float));
    hipMemcpy( device_result, &host_result, sizeof(float), hipMemcpyHostToDevice);

    sdot_kernel<<<(n+255)/256, 256>>>(n, device_x, device_y, device_result);
    
    hipMemcpy(&host_result, device_result, sizeof(float), hipMemcpyDeviceToHost);

    hipFree(device_x);
    hipFree(device_y);
    hipFree(device_result);

    return host_result;
}
