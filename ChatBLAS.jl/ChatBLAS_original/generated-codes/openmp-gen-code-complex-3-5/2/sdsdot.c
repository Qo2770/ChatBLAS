#include "chatblas_openmp.h"

float chatblas_sdsdot(int n, float b, float *x, float *y) {
    float result = b;

    #pragma omp parallel for reduction(+:result)
    for (int i = 0; i < n; i++) {
        result += (float) x[i] * (float) y[i];
    }

    return result;
}