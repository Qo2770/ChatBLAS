#include "chatblas_openmp.h"

void chatblas_scopy(int n, float *x, float *y) {
    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < n; i++)
        y[i] = x[i];
}