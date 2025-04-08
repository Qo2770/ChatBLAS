#include "chatblas_openmp.h"
#include <omp.h>

float chatblas_sasum(int n, float *x) {
    float sum = 0.0f;
    int i;

    #pragma omp parallel for reduction(+:sum)
    for (i = 0; i < n; i++) {
        sum += fabsf(x[i]);
    }

    return sum;
}