#include "chatblas_openmp.h"
#include <omp.h>

float chatblas_sdsdot(int n, float b, float *x, float *y) {
    double dot_product = 0.0;

    // Parallelize the loop to compute the dot product using OpenMP
    #pragma omp parallel for reduction(+:dot_product)
    for (int i = 0; i < n; i++) {
        // Cast elements to double for double precision in the computation
        double xi = (double)x[i];
        double yi = (double)y[i];

        dot_product += xi * yi;
    }

    // Add scalar b after converting it to double and return the result as float
    dot_product += (double)b;

    // Convert the result back to float
    return (float)dot_product;
}