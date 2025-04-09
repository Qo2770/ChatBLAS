#include "chatblas_openmp.h"
#include <omp.h>

void chatblas_sscal(int n, float a, float *x) {
    // Ensure there's something to scale
    if (n <= 0 || x == NULL) return;

    // Set the number of threads
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        x[i] *= a;
    }
}