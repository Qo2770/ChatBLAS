#include "chatblas_openmp.h"
#include <omp.h>

void chatblas_scopy(int n, float *x, float *y) {
    #pragma omp parallel
    {
        int i;
        #pragma omp for
        for(i=0; i<n; i++)
            y[i] = x[i];
    }
}
