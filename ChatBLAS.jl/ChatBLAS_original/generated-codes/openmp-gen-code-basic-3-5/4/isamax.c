#include "chatblas_openmp.h"

int chatblas_isamax(int n, float *x) {
    int i, max_pos;
    float max_val;

    #pragma omp parallel for
    for (i = 0; i < n; i++) {
        #pragma omp critical
        {
            if (i == 0 || fabsf(x[i]) > max_val) {
                max_val = fabsf(x[i]);
                max_pos = i;
            }
        }
    }

    return max_pos;
}
