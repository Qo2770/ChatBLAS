#include "chatblas_openmp.h"
#include <omp.h>

float chatblas_sdsdot(int n, float b, float *x, float *y) {
    double dot_product = 0.0;

    // Parallelize the loop to calculate the dot product
    #pragma omp parallel for reduction(+:dot_product)
    for (int i = 0; i < n; i++) {
        double x_double = (double)x[i]; // Cast elements of x to double
        double y_double = (double)y[i]; // Cast elements of y to double
        dot_product += x_double * y_double;
    }

    // Add the scalar b to the dot product and return as float
    return (float)(dot_product + (double)b);
}