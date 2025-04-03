#include "chatblas_cuda.h"

float chatblas_sdsdot(int n, float b, float *x, float *y) {
    double acc = 0.0;

    #pragma omp parallel for reduction(+: acc)
    for (int i = 0; i < n; i++) {
        float xi = (float) x[i];
        float yi = (float) y[i];
        acc += xi * yi;
    }

    double result = acc + (double) b;
    return (float) result;
}
