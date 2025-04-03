#include "chatblas_openmp.h"

int chatblas_isamax(int n, float *x) {
    int index = 0;
    float max_val = fabs(x[0]);

    #pragma omp parallel for
    for(int i = 1; i < n; i++) {
        if(fabs(x[i]) > max_val) {
            #pragma omp critical
            {
                if(fabs(x[i]) > max_val) {
                    max_val = fabs(x[i]);
                    index = i;
                }
            }
        }
    }
    
    return index;
}