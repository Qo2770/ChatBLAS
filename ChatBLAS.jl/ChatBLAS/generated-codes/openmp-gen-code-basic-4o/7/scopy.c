#include "chatblas_openmp.h"
#include <omp.h>

void chatblas_scopy(int n, float *x, float *y) {
    // Use OpenMP to parallelize the loop
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        y[i] = x[i];
    }
}