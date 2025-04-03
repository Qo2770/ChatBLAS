#include "chatblas_openmp.h"

float chatblas_snrm2(int n, float *x) {
    float normSquared = 0.0;

    #pragma omp parallel for reduction(+:normSquared)
    for (int i = 0; i < n; ++i) {
        normSquared += x[i] * x[i];
    }

    return sqrt(normSquared);
}
