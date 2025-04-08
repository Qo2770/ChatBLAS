#include "chatblas_openmp.h"
#include <omp.h>

void chatblas_scopy(int n, float *x, float *y) {
    // Ensure that both x and y pointers are non-null
    if (x == NULL || y == NULL) {
        return;
    }

    // Use OpenMP to parallelize the for loop
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        y[i] = x[i];
    }
}