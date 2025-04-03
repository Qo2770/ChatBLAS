#include "chatblas_openmp.h"

int chatblas_isamax(int n, float *x) {
    int max_pos = 0;
    float max_val = 0;
    
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        if (fabs(x[i]) > max_val) {
            max_val = fabs(x[i]);
            max_pos = i;
        }
    }
    
    return max_pos + 1;
}