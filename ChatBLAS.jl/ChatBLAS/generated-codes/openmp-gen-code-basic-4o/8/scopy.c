#include "chatblas_openmp.h"
#include <omp.h>

void chatblas_scopy(int n, float *x, float *y) {
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        y[i] = x[i];
    }
}