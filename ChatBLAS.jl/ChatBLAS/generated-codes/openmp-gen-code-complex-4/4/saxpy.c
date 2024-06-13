#include "chatblas_openmp.h"

void chatblas_saxpy(int n, float a, float *x, float *y) {
    int i;
    #pragma omp parallel for
    for(i = 0; i < n; i++) {
        y[i] = a*x[i] + y[i];
    }
}