#include "chatblas_hip.h"

__global__ void isamax_kernel(int n, float *x, int *ind) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  __shared__ float local_max_val[256];
  __shared__ int local_max_id[256];

  local_max_val[threadIdx.x] = 0;
  local_max_id[threadIdx.x] = -1;

  if(idx < n) {
    local_max_val[threadIdx.x] = fabs(x[idx]);
    local_max_id[threadIdx.x] = idx;
  }

  __syncthreads();

  for(int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if(threadIdx.x < stride && local_max_val[threadIdx.x + stride] > local_max_val[threadIdx.x]) {
      local_max_val[threadIdx.x] = local_max_val[threadIdx.x + stride];
      local_max_id[threadIdx.x] = local_max_id[threadIdx.x + stride];
    }
    __syncthreads();
  }

  if(threadIdx.x == 0) {
    ind[blockIdx.x] = local_max_id[0];
  }
}

int chatblas_isamax(int n, float *x) {
  float *d_x;
  int *d_ind;
  int numBlocks = (n + 255) / 256;
  int *h_ind = (int*)malloc(numBlocks * sizeof(int));

  hipMalloc((void**)&d_x, n * sizeof(float));
  hipMalloc((void**)&d_ind, numBlocks * sizeof(int));

  hipMemcpy(d_x, x, n * sizeof(float), hipMemcpyHostToDevice);

  isamax_kernel<<<numBlocks, 256>>>(n, d_x, d_ind);

  hipMemcpy(h_ind, d_ind, numBlocks * sizeof(int), hipMemcpyDeviceToHost);

  int max_id = 0;
  for(int i = 0; i < numBlocks; i++) {
    if(fabs(x[h_ind[i]]) > fabs(x[max_id])) {
      max_id = h_ind[i];
    }
  }

  hipFree(d_x);
  hipFree(d_ind);
  free(h_ind);

  return max_id;
}
