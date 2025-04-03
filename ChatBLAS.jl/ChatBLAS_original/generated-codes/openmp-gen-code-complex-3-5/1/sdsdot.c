#include "chatblas_openmp.h"

float chatblas_sdsdot(int n, float b, float *x, float *y) {
    float dot_product = 0.0;
    
    #pragma omp parallel for reduction(+:dot_product)
    for (int i = 0; i < n; i++) {
        float xi = (float)x[i];
        float yi = (float)y[i];
        dot_product += xi * yi;
    }
    
    dot_product += b;
    
    return dot_product;
}