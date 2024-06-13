#include <omp.h>
#include "chatblas_openmp.h"

int chatblas_isamax(int n, float *x) {
    int idx_max = 0;
    float max_val = 0.0f;

    #pragma omp parallel for
    for (int i = 0; i < n; i+=2) {
        float temp_val = (x[i] > 0) ? x[i] : -x[i];
        if (temp_val > max_val) {
            #pragma omp critical
            {
                if (temp_val > max_val) {
                    max_val = temp_val;
                    idx_max = i+1;
                }
            }
        }
    }
    
    return idx_max;
}
