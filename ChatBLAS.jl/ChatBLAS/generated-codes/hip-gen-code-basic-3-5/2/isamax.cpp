__global__ void isamax_kernel(int n, float *x, int *ind) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    __shared__ int max_ind;
    __shared__ float max_val;
    
    if (tid == 0) {
        max_ind = 0;
        max_val = fabsf(x[0]);
    }
    
    __syncthreads();
    
    for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
        float val = fabsf(x[i]);
        if (val > max_val) {
            max_ind = i;
            max_val = val;
        }
    }
    
    __syncthreads();
    
    if (tid == 0) {
        *ind = max_ind;
    }
}

int chatblas_isamax(int n, float *x) {
    float *d_x;
    int *d_ind;
    
    hipMalloc(&d_x, n * sizeof(float));
    hipMalloc(&d_ind, sizeof(int));
    
    hipMemcpy(d_x, x, n * sizeof(float), hipMemcpyHostToDevice);
    
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    
    isamax_kernel<<<grid_size, block_size>>>(n, d_x, d_ind);
    
    int ind;
    hipMemcpy(&ind, d_ind, sizeof(int), hipMemcpyDeviceToHost);
    
    hipFree(d_x);
    hipFree(d_ind);
    
    return ind;
}