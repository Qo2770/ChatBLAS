__global__ void isamax_kernel(int n, float *x, int *ind) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n) {
        if (fabsf(x[tid]) > fabsf(x[ind[0]])) {
            ind[0] = tid;
        }
    }
}

int chatblas_isamax(int n, float *x) {
    int *d_ind;
    float *d_x;

    hipMalloc(&d_ind, sizeof(int));
    hipMalloc(&d_x, n * sizeof(float));

    hipMemcpy(d_x, x, n * sizeof(float), hipMemcpyHostToDevice);

    int initial_index = 0;
    hipMemcpy(d_ind, &initial_index, sizeof(int), hipMemcpyHostToDevice);

    dim3 blockSize(256);
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x);

    isamax_kernel<<<gridSize, blockSize>>>(n, d_x, d_ind);

    int result;
    hipMemcpy(&result, d_ind, sizeof(int), hipMemcpyDeviceToHost);

    hipFree(d_x);
    hipFree(d_ind);

    return result;
}
