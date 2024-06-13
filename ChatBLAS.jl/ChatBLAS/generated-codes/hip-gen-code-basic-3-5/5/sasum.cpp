__global__ void sasum_kernel(int n, float *x, float *sum) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n) {
        sum[tid] = fabs(x[tid]);
    }
}

float chatblas_sasum(int n, float *x) {
    float *d_x, *d_sum;
    float h_sum[n];
    
    hipMalloc(&d_x, n * sizeof(float));
    hipMalloc(&d_sum, n * sizeof(float));
    
    hipMemcpy(d_x, x, n * sizeof(float), hipMemcpyHostToDevice);
    
    sasum_kernel<<<(n+255)/256, 256>>>(n, d_x, d_sum);
    
    hipMemcpy(h_sum, d_sum, n * sizeof(float), hipMemcpyDeviceToHost);
    
    float result = 0;
    for (int i = 0; i < n; i++) {
        result += h_sum[i];
    }
    
    hipFree(d_x);
    hipFree(d_sum);
    
    return result;
}