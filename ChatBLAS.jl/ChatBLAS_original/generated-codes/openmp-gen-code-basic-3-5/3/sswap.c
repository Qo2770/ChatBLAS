#include "chatblas_openmp.h"
#include <omp.h>

void chatblas_sswap(int n, float *x, float *y) {
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        float temp = x[i];
        x[i] = y[i];
        y[i] = temp;
    }
}
