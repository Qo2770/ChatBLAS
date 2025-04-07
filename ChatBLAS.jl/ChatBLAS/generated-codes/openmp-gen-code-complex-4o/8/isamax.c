#include "chatblas_openmp.h"
int chatblas_isamax(int n, float *x) {
    int index = 0;
    float max_val = 0.0f;
    #pragma omp parallel
    {
        int local_index = 0;
        float local_max_val = 0.0f;
        #pragma omp for
        for (int i = 0; i < n; i++) {
            float abs_val = fabsf(x[i]);
            if (abs_val > local_max_val) {
                local_max_val = abs_val;
                local_index = i;
            }
        }
        #pragma omp critical
        {
            if (local_max_val > max_val) {
                max_val = local_max_val;
                index = local_index;
            }
        }
    }
    return index;
}