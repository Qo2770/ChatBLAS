#include "chatblas_openmp.h"

float chatblas_sdot(int n, float *x, float *y) {
    float dot_product = 0.0;

    #pragma omp parallel for reduction(+:dot_product)
    for (int i = 0; i < n; i++) {
        dot_product += x[i] * y[i];
    }

    return dot_product;
}