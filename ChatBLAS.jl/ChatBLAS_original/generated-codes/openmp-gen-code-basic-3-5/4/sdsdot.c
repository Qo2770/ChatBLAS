#include "chatblas_openmp.h"

float chatblas_sdsdot(int n, float b, float *x, float *y) {
  double result = 0.0;

  #pragma omp parallel for reduction(+:result)
  for (int i = 0; i < n; i++) {
    double xi = (double)x[i];
    double yi = (double)y[i];
    result += xi * yi;
  }

  result += (double)b;

  return (float)result;
}