#include "chatblas_openmp.h"
#include <omp.h>

float chatblas_sdsdot(int n, float b, float *x, float *y) {
    double result = 0.0;

    #pragma omp parallel for reduction(+:result)
    for (int i = 0; i < n; i++) {
        result += (double)x[i] * (double)y[i];
    }

    // Adding scalar b to the dot product
    result += b;

    return (float)result;
}
