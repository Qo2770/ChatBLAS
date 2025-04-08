#include "chatblas_openmp.h"
int chatblas_isamax(int n, float *x) {
    int index = 0;
    float max_val = 0.0f;
    #pragma omp parallel
    {
        float local_max = 0.0f;
        int local_index = 0;
        #pragma omp for
        for (int i = 0; i < n; i++) {
            float abs_val = fabs(x[i]);
            if (abs_val > local_max) {
                local_max = abs_val;
                local_index = i;
            }
        }
        #pragma omp critical
        {
            if (local_max > max_val) {
                max_val = local_max;
                index = local_index;
            }
        }
    }
    return index;
}