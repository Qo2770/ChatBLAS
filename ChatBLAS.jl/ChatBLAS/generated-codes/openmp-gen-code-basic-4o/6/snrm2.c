#include "chatblas_openmp.h"
#include <math.h>
#include <omp.h>

float chatblas_snrm2(int n, float *x) {
    float sum = 0.0f;
    
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < n; i++) {
        sum += x[i] * x[i];
    }
    
    return sqrtf(sum);
}