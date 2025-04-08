#include "chatblas_openmp.h"
#include <omp.h>

int chatblas_isamax(int n, float *x) {
    int max_index = 0;
    float max_value = fabsf(x[0]);

    #pragma omp parallel
    {
        int local_index = 0;
        float local_max = fabsf(x[0]);
        
        #pragma omp for
        for (int i = 1; i < n; i++) {
            float abs_value = fabsf(x[i]);
            if (abs_value > local_max) {
                local_max = abs_value;
                local_index = i;
            }
        }
        
        #pragma omp critical
        {
            if (local_max > max_value) {
                max_value = local_max;
                max_index = local_index;
            }
        }
    }
    
    return max_index;
}