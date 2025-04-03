__global__ void isamax_kernel(int n, float *x, int *ind) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n) {
        if (abs(x[tid]) > atomicMax(ind, abs(x[tid]))) {
            *ind = tid;
        }
    }
}

int chatblas_isamax(int n, float *x) {
    float *d_x;
    int *d_ind;
    int h_ind;
    
    hipMalloc(&d_x, n * sizeof(float));
    hipMemcpy(d_x, x, n * sizeof(float), hipMemcpyHostToDevice);
    
    hipMalloc(&d_ind, sizeof(int));
    
    hipLaunchKernelGGL(isamax_kernel, dim3((n+255)/256), dim3(256), 0, 0, n, d_x, d_ind);
    
    hipMemcpy(&h_ind, d_ind, sizeof(int), hipMemcpyDeviceToHost);
    
    hipFree(d_x);
    hipFree(d_ind);
    
    return h_ind;
}