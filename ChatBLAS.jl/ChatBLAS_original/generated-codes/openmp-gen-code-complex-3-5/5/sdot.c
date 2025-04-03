#include "chatblas_openmp.h"

float chatblas_sdot(int n, float *x, float *y) {
    float sum = 0.0;
    int i;

    #pragma omp parallel for reduction(+:sum)
    for (i = 0; i < n; i++) {
        sum += x[i] * y[i];
    }

    return sum;
}