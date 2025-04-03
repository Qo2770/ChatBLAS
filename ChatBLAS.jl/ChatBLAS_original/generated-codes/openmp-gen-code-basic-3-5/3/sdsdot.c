#include "chatblas_openmp.h"

float chatblas_sdsdot(int n, float b, float *x, float *y) {
    double dot_product = 0.0;

    #pragma omp parallel for reduction(+:dot_product)
    for (int i = 0; i < n; ++i) {
        float x_casted = (float)x[i];
        float y_casted = (float)y[i];
        dot_product += (double)(x_casted * y_casted);
    }

    dot_product += (double)b;

    return (float)dot_product;
}
