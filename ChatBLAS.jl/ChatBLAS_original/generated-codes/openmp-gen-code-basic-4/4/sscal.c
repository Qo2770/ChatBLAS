#include "chatblas_openmp.h"
#include <omp.h>

void chatblas_sscal(int n, float a, float *x) {
    int i;

    #pragma omp parallel for
    for (i=0; i<n; i++) {
        x[i] = a * x[i];
    }
}
