#include "chatblas_openmp.h"

float chatblas_sdot(int n, float *x, float *y) {
    float dotProduct = 0.0;
    
    #pragma omp parallel for reduction(+:dotProduct)
    for (int i = 0; i < n; i++) {
        dotProduct += x[i] * y[i];
    }
    
    return dotProduct;
}
