#include "chatblas_openmp.h"

int chatblas_isamax(int n, float *x) {
    int index = 0;
    float max_val = 0.0;

    #pragma omp parallel for
    for(int i = 0; i < n; i++) {
        float abs_val = abs(x[i]);
        #pragma omp critical
        {
            if(abs_val > max_val) {
                max_val = abs_val;
                index = i;
            }
        }
    }
    
    return index;
}