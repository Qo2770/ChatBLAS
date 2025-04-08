#include "chatblas_openmp.h"
#include <omp.h>

void chatblas_sscal(int n, float a, float *x) {
    // Define the number of threads to use.
    // You can adjust this value or control it via environment variables.
    // int num_threads = omp_get_max_threads();

    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        x[i] *= a;
    }
}