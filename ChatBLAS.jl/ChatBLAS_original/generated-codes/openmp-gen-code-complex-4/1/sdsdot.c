#include "chatblas_openmp.h"

float chatblas_sdsdot(int n, float b, float *x, float *y) { 
    float result = 0.0; 
    #pragma omp parallel for reduction(+:result) 
    for (int i = 0; i < n; i++) { 
        result += ((double)x[i] * (double)y[i]); 
    } 
    return result + b; 
}