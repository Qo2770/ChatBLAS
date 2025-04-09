#include "chatblas_openmp.h"
#include <omp.h>

void chatblas_scopy(int n, float *x, float *y) {
    // Set the number of threads you want to use
    int num_threads = omp_get_max_threads();

    // Use OpenMP parallel for loop to copy elements
    #pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < n; i++) {
        y[i] = x[i];
    }
}