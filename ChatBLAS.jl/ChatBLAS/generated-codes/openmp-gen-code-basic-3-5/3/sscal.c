#include <stdio.h>
#include <omp.h>
#include "chatblas_openmp.h"

void chatblas_sscal(int n, float a, float *x) {
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        x[i] *= a;
    }
}

int main() {
    int n = 5;
    float a = 2.0;
    float x[] = {1.0, 2.0, 3.0, 4.0, 5.0};

    printf("Vector before scaling:\n");
    for (int i = 0; i < n; i++) {
        printf("%0.1f ", x[i]);
    }
    printf("\n");

    chatblas_sscal(n, a, x);

    printf("Vector after scaling:\n");
    for (int i = 0; i < n; i++) {
        printf("%0.1f ", x[i]);
    }
    printf("\n");

    return 0;
}
