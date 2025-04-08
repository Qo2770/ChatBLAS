#include "chatblas_openmp.h"
#include <omp.h>
#include <math.h>

int chatblas_isamax(int n, float *x) {
    int index_of_max = 0;
    float max_value = 0.0f;

    #pragma omp parallel
    {
        int local_index_of_max = 0;
        float local_max_value = 0.0f;

        #pragma omp for
        for (int i = 0; i < n; i++) {
            float abs_value = fabs(x[i]);
            if (abs_value > local_max_value) {
                local_max_value = abs_value;
                local_index_of_max = i;
            }
        }

        #pragma omp critical
        {
            if (local_max_value > max_value) {
                max_value = local_max_value;
                index_of_max = local_index_of_max;
            }
        }
    }
    return index_of_max;
}