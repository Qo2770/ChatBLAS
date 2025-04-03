#include "chatblas_openmp.h"
#include <omp.h>

float chatblas_sdot(int n, float *x, float *y) {
  float result = 0.0;
  #pragma omp parallel for reduction(+:result)
    for(int i=0; i<n; i++) {
      result += x[i] * y[i];
    }

    return result;
}
