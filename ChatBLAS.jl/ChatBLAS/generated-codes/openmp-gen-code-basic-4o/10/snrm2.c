#include "chatblas_openmp.h"
#include <math.h>

float chatblas_snrm2(int n, float *x) {
    float norm = 0.0f;
    
    #pragma omp parallel for reduction(+:norm)
    for (int i = 0; i < n; i++) {
        norm += x[i] * x[i];
    }
    
    return sqrtf(norm);
}