#include "chatblas_openmp.h"

void chatblas_sswap(int n, float *x, float *y) {
    int i;
    float temp;
    #pragma omp parallel for private(i, temp)
    for (i = 0; i < n; i++){
        temp = x[i];
        x[i] = y[i];
        y[i] = temp;
    }
}