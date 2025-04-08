#include "chatblas_openmp.h"
#include <omp.h> // Include the OpenMP header
#include <stddef.h> // For size_t

void chatblas_sscal(int n, float a, float *x) {
    // Parallelize the loop using OpenMP
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        x[i] *= a;
    }
}