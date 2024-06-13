#include "chatblas_openmp.h"

void chatblas_sswap(int n, float *x, float *y) {
    int i;
    #pragma omp parallel for default(none) shared(x,y,n) private(i)
    for(i=0; i<n; i++){
        float temp = x[i];
        x[i] = y[i];
        y[i] = temp;
    }
}