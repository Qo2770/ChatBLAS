#include "chatblas_openmp.h"

void chatblas_sscal( int n, float a , float *x) {
    #pragma omp parallel for
    for(int i = 0; i < n; i++) {
        x[i] = a * x[i];
    }
}