#include "chatblas_openmp.h"

float chatblas_sasum(int n, float *x) {
    float sum = 0.0f;
    
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < n; ++i) {
        sum += fabsf(x[i]);
    }
    
    return sum;
}
