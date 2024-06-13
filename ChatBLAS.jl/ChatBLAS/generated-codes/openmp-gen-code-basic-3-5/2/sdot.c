#include "chatblas_openmp.h"
#include <omp.h>

float chatblas_sdot(int n, float *x, float *y) {
    float dot_product = 0.0;
    omp_set_num_threads(omp_get_max_threads());

    #pragma omp parallel for reduction(+:dot_product)
    for (int i = 0; i < n; i++) {
        dot_product += x[i] * y[i];
    }

    return dot_product;
}