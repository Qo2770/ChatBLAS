__global__ void sasum_kernel(int n, float *x, float *sum) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n) {
        atomicAdd(sum, fabs(x[tid]));
    }
}

float chatblas_sasum(int n, float *x) {
    float *d_x, *d_sum, sum = 0;
    
    hipMalloc(&d_x, n * sizeof(float));
    hipMalloc(&d_sum, sizeof(float));
    
    hipMemcpy(d_x, x, n * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_sum, &sum, sizeof(float), hipMemcpyHostToDevice);
    
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    
    sasum_kernel<<<grid_size, block_size>>>(n, d_x, d_sum);
    
    hipMemcpy(&sum, d_sum, sizeof(float), hipMemcpyDeviceToHost);
    
    hipFree(d_x);
    hipFree(d_sum);
    
    return sum;
}