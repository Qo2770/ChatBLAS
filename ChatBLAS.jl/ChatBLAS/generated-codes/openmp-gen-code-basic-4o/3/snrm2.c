#include "chatblas_openmp.h"
#include <math.h>
#include <omp.h>

float chatblas_snrm2(int n, float *x) {
    float norm = 0.0f;
    #pragma omp parallel
    {
        float sum = 0.0f;
        #pragma omp for
        for (int i = 0; i < n; i++) {
            sum += x[i] * x[i];
        }
        #pragma omp atomic
        norm += sum;
    }
    return sqrtf(norm);
}