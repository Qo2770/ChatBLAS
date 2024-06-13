#include "chatblas_openmp.h"

float chatblas_snrm2(int n, float *x) {
    float norm = 0.0;
    int i;
    #pragma omp parallel for reduction(+:norm)
    for (i = 0; i < n; i++) {
        norm += x[i]*x[i];
    }
    return sqrt(norm);
}