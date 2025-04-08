#include "chatblas_openmp.h"
#include <omp.h>

void chatblas_saxpy(int n, float a, float *x, float *y) {
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        y[i] = a * x[i] + y[i];
    }
}