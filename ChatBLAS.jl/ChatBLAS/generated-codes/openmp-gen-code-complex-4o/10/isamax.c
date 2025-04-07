#include "chatblas_openmp.h"
int chatblas_isamax(int n, float *x) {
    int max_index = 0;
    float max_value = 0.0f;
    #pragma omp parallel
    {
        int local_max_index = 0;
        float local_max_value = 0.0f;
        #pragma omp for
        for (int i = 0; i < n; i++) {
            if (fabsf(x[i]) > local_max_value) {
                local_max_value = fabsf(x[i]);
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