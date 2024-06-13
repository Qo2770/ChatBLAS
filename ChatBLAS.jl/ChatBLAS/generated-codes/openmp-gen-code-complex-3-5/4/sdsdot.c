#include "chatblas_openmp.h"

float chatblas_sdsdot(int n, float b, float *x, float *y) {
    float sum = 0.0;
    #pragma omp parallel for shared(sum)
    for (int i = 0; i < n; i++) {
        float xi = (float)x[i];
        float yi = (float)y[i];
        float dot_product = xi * yi;
        #pragma omp atomic
        sum += dot_product;
    }
    sum += b;
    return sum;
}