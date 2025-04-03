#include "chatblas_openmp.h"

float chatblas_snrm2(int n, float *x) {
    int i;
    float norm = 0.0;

    #pragma omp parallel for reduction(+:norm)
    for (i = 0; i < n; i++) {
        norm += x[i] * x[i];
    }

    return sqrt(norm);
}