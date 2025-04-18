#include "chatblas_openmp.h"
#include <omp.h>

float chatblas_sasum(int n, float *x) {
    float sum = 0.0;
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < n; i++) {
        sum += fabs(x[i]);
    }
    return sum;
}