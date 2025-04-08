#include "chatblas_openmp.h"
#include <stdlib.h>
#include <omp.h>
#include <math.h>

int chatblas_isamax(int n, float *x) {
    int max_index = 0;
    float max_value = fabsf(x[0]);

    #pragma omp parallel
    {
        int local_max_index = 0;
        float local_max_value = max_value;
        
        #pragma omp for
        for (int i = 1; i < n; i++) {
            float abs_value = fabsf(x[i]);
            if (abs_value > local_max_value) {
                local_max_value = abs_value;
                local_max_index = i;
            }
        }

        #pragma omp critical
        {
            if (local_max_value > max_value) {
                max_value = local_max_value;
                max_index = local_max_index;
            }
        }
    }

    return max_index;
}