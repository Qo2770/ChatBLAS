#include "chatblas_openmp.h"
#include <omp.h>

float chatblas_sdsdot(int n, float b, float *x, float *y) {
    double sum = 0.0;

    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < n; i++) {
        sum += (double)x[i] * (double)y[i];
    }

    // Add scalar b to the dot product result
    sum += (double)b;

    // Cast back to float
    return (float)sum;
}