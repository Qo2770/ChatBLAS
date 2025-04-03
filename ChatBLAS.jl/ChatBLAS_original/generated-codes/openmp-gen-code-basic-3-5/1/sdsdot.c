#include "chatblas_openmp.h"

float chatblas_sdsdot(int n, float b, float *x, float *y) {
    double dot = 0.0;
    
    #pragma omp parallel for reduction(+:dot)
    for (int i = 0; i < n; i++) {
        float x_casted = (float)x[i];
        float y_casted = (float)y[i];
        dot += (double)(x_casted * y_casted);
    }
    
    dot += (double)b;
    
    return (float)dot;
}