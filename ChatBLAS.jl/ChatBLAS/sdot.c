#include "chatblas_openmp.h"
#include <stdio.h>

float chatblas_sdot(int n, float *x, float *y) {
    float result = 0.0; 
    int i;
    #pragma omp parallel for reduction(+:result)
    for (i = 0; i < n; i++) {
        result += x[i] * y[i];
    }
    return result;
}
