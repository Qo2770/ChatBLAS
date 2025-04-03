#include "chatblas_openmp.h"
#include <omp.h>

int chatblas_isamax(int n, float *x) {
  int max_index = 0;
  float max_value = 0.0;
  
  #pragma omp parallel for
  for (int i = 0; i < n; i++) {
    if (fabs(x[i]) > max_value) {
      #pragma omp critical
      {
        if (fabs(x[i]) > max_value) {
          max_index = i;
          max_value = fabs(x[i]);
        }
      }
    }
  }
  
  return max_index;
}