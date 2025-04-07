#include "chatblas_openmp.h"
int chatblas_isamax(int n, float *x) {
    int max_index = 0;
    float max_value = 0.0;
    #pragma omp parallel
    {
        int index_local = 0;
        float value_local = 0.0;
        #pragma omp for
        for (int i = 0; i < n; i++) {
            if (fabsf(x[i]) > value_local) {
                value_local = fabsf(x[i]);
                index_local = i;
            }
        }
        #pragma omp critical
        {
            if (value_local > max_value) {
                max_value = value_local;
                max_index = index_local;
            }
        }
    }
    return max_index;
}