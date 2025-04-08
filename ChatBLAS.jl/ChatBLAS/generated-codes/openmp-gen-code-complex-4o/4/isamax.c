#include "chatblas_openmp.h"

int chatblas_isamax(int n, float *x) {
    int index = 0;
    float max_abs_val = 0.0;
    
    #pragma omp parallel
    {
        int local_index = 0;
        float local_max_abs_val = 0.0;
        
        #pragma omp for
        for (int i = 0; i < n; ++i) {
            float abs_val = fabs(x[i]);
            if (abs_val > local_max_abs_val) {
                local_max_abs_val = abs_val;
                local_index = i;
            }
        }

        #pragma omp critical
        {
            if (local_max_abs_val > max_abs_val) {
                max_abs_val = local_max_abs_val;
                index = local_index;
            }
        }
    }
    
    return index;
}