#include "chatblas_openmp.h"
#include <omp.h>

float chatblas_sdsdot(int n, float b, float *x, float *y) {
    double dot_product = 0.0; // Use double precision for accumulation

    // Parallelize with OpenMP
    #pragma omp parallel for reduction(+:dot_product)
    for (int i = 0; i < n; i++) {
        dot_product += (double)x[i] * (double)y[i];
    }

    // Add the scalar b and return the result as a float
    return (float)(dot_product + (double)b);
}