#include <stdio.h>
#include "chatblas_openmp.h"

float chatblas_sasum(int n, float *x) {
    float sum = 0.0;
    
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < n; i++) {
        sum += (x[i] >= 0) ? x[i] : -x[i];
    }
    
    return sum;
}

int main() {
    int n = 5;
    float x[] = {-1.5, 2.5, -3.0, 4.0, 5.5};
    
    float result = chatblas_sasum(n, x);
    printf("Sum of absolute values: %.2f\n", result);
    
    return 0;
}
