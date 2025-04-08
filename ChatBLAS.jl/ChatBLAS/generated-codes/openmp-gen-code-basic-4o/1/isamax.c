#include "chatblas_openmp.h"
#include <omp.h>

int chatblas_isamax(int n, float *x) {
    int max_index = 0;
    float max_value = 0.0f;

    #pragma omp parallel
    {
        int local_max_index = 0;
        float local_max_value = 0.0f;

        #pragma omp for nowait
        for (int i = 0; i < n; i++) {
            float abs_value = fabs(x[i]);
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