#include "chatblas_openmp.h"
float chatblas_snrm2(int n, float *x) {
    float sum = 0.0;
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < n; i++) {
        sum += x[i] * x[i];
    }
    return sqrt(sum);
}