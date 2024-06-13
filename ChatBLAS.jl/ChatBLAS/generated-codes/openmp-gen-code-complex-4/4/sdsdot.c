#include "chatblas_openmp.h"

float chatblas_sdsdot(int n, float b, float *x, float *y) {
    double sum = 0.0;
    int i;

    #pragma omp parallel for private(i) reduction(+:sum)
    for(i = 0; i < n; i++) {
        sum += ((double)x[i]) * ((double)y[i]);
    }
    
    return (float)(sum + b);
}