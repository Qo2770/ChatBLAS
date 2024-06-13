#include "chatblas_openmp.h"

float chatblas_sasum(int n, float *x) {
    int i;
    float sum = 0.0;

    #pragma omp parallel for reduction(+:sum)
    for (i = 0; i < n; i++) {
        sum += fabs(x[i]);
    }

    return sum;
}
