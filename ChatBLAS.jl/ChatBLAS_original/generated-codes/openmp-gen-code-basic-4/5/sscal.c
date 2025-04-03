#include "chatblas_openmp.h"

void chatblas_sscal( int n, float a , float *x) {
    //Variables for parallel loop
    int i;

    //Parallel loop using OpenMP
    #pragma omp parallel for
    for(i=0; i<n; i++) {
        x[i] = a * x[i];
    }
}
