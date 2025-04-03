#include "chatblas_openmp.h"

float chatblas_snrm2(int n, float *x) {
    float result = 0.0f;
    int i;

    #pragma omp parallel for reduction(+:result)
    for(i = 0; i < n; i++){
        result += x[i]*x[i];
    }

    return sqrt(result);
}