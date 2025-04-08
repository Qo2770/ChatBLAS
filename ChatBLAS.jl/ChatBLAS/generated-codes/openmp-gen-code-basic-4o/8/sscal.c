#include "chatblas_openmp.h"
#include <omp.h>  // Include OpenMP header

void chatblas_sscal(int n, float a, float *x) {
    // Use OpenMP to parallelize the scaling operation
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        x[i] *= a;
    }
}