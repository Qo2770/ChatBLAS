__global__ void sdsdot_kernel( int n, float b, float *x, float *y, double *res ) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ double tmp[256];

    double sum = 0.0;
    for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
        sum += (double)x[i] * (double)y[i];
    }
    tmp[threadIdx.x] = sum;

    __syncthreads();

    for (int s = 1; s < blockDim.x; s *= 2) {
        int index = 2 * s * threadIdx.x;
        if (index < blockDim.x) {
            tmp[index] += tmp[index + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(res, tmp[0] + (double)n * (double)b);
    }
}

float chatblas_sdsdot( int n, float b, float *x, float *y) {
    float *d_x, *d_y;
    double *d_res;
    double h_res = 0.0;

    hipMalloc((void **)&d_x, n * sizeof(float));
    hipMalloc((void **)&d_y, n * sizeof(float));
    hipMalloc((void **)&d_res, sizeof(double));

    hipMemcpy(d_x, x, n * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_y, y, n * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_res, &h_res, sizeof(double), hipMemcpyHostToDevice);

    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;

    sdsdot_kernel<<<grid_size, block_size>>>(n, b, d_x, d_y, d_res);

    hipMemcpy(&h_res, d_res, sizeof(double), hipMemcpyDeviceToHost);

    hipFree(d_x);
    hipFree(d_y);
    hipFree(d_res);

    return (float)h_res;
}
