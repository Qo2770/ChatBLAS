#include "chatblas_openmp.h"
#include <omp.h>
#include <stdio.h>

float chatblas_sdsdot(int n, float b, float *x, float *y) {
    double dot_product = 0.0;  // Use double for accumulation

    #pragma omp parallel for reduction(+:dot_product)
    for (int i = 0; i < n; i++) {
        dot_product += (double)x[i] * (double)y[i];
    }

    // Add the scalar b to the dot product
    dot_product += (double)b;

    return (float)dot_product;
}

int main() {
    // Example usage in main
    int n = 4;
    float x[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float y[] = {5.0f, 6.0f, 7.0f, 8.0f};
    float b = 1.5f;

    float result = chatblas_sdsdot(n, b, x, y);
    printf("Result: %f\n", result);

    return 0;
}