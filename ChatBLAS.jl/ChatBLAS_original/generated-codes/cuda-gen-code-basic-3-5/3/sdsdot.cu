#include <chatblas_cuda.h>
#include <omp.h>

float chatblas_sdsdot(int n, float b, float *x, float *y) {

    double dot = 0.0;

    #pragma omp parallel for reduction(+:dot)
    for(int i = 0; i < n; i++) {
        dot += (double)((float)x[i]) * ((float)y[i]);
    }

    return (float)(dot) + b;
}