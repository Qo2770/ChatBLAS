#include "chatblas_openmp.h"

float chatblas_sdot(int n, float *x, float *y) {
    float sum = 0.0;

    #pragma omp parallel for reduction(+:sum)
    for(int i=0; i<n; i++){
        sum += x[i] * y[i];
    }

    return sum;
}
