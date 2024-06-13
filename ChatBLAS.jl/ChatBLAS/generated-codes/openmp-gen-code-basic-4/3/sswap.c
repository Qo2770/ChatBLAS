#include "chatblas_openmp.h"

void chatblas_sswap(int n, float *x, float *y) {
  #pragma omp parallel for
  for (int i = 0; i < n; i++) {
      float tempSwap = x[i];
      x[i] = y[i];
      y[i] = tempSwap;
  }
}
