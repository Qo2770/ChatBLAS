#include "chatblas_openmp.h"
#include <math.h>

int chatblas_isamax(int n, float *x) {
    int i, max_idx = 0;
    float max_value = fabs(x[0]);

    #pragma omp parallel for
    for (i = 1; i < n; i++) {
        float absolute_value = fabs(x[i]);

        #pragma omp critical
        {
            if (absolute_value > max_value) {
                max_value = absolute_value;
                max_idx = i;
            }
        }
    }
    
    return max_idx;
}
