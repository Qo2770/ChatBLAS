#include "chatblas_openmp.h"

float chatblas_sdsdot(int n, float b, float *x, float *y) {
    double result = 0.0;
    int i;

    #pragma omp parallel for reduction(+:result)
    for (i = 0; i < n; i++) {
        result += (double)x[i] * (double)y[i];
    }

    return (float)result + b;
}