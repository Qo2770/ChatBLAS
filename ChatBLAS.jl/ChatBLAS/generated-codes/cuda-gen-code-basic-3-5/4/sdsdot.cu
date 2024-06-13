#include "chatblas_cuda.h"

float chatblas_sdsdot(int n, float b, float *x, float *y) {
    double dotProduct = 0.0;

    #pragma omp parallel for reduction(+:dotProduct)
    for (int i = 0; i < n; i++) {
        float X = (float) x[i];
        float Y = (float) y[i];
        dotProduct += (double) (X * Y);
    }

    return (float) (dotProduct + b);
}
